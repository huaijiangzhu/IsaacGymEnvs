import torch
if torch.__version__ >= '2.0.0':
    from torch import func as functorch
else:
    import functorch

class QP(object):
    @torch.no_grad()
    def __init__(self, num_batches, num_vars, num_eqc, friction_coeff=None, device=torch.device("cuda:0")):
        self.device = device
        
        # problem params
        self.num_vars = num_vars
        self.num_batches = num_batches
        self.num_eqc = num_eqc

        if isinstance(friction_coeff, torch.Tensor):
            self.friction_coeff = friction_coeff.to(device).view(num_batches, 1)
        elif isinstance(friction_coeff, float):
            self.friction_coeff = friction_coeff * torch.ones(num_batches, 1).to(device)
        else:
            self.friction_coeff = None
        
        # internal states
        self.xk = torch.zeros(self.num_batches, self.num_vars).to(self.device)
        self.yk = torch.zeros(self.num_batches, self.num_vars).to(self.device)
        # # if it's a force problem, initialize the forces as the unit contact normal
        # self.xk[:, 2:3] = 1.0
        # self.yk[:, 2:3] = 1.0
        self.xk1 = torch.zeros(self.num_batches, self.num_vars).to(self.device)
        self.yk1 = torch.zeros(self.num_batches, self.num_vars).to(self.device)
        self.zk = torch.zeros(self.num_batches, self.num_vars).to(self.device)
        self.grad = torch.zeros(self.num_batches, self.num_vars).to(self.device)

        self.Pk = torch.zeros(self.num_batches, self.num_eqc).to(self.device)
        self.ydiff = torch.zeros(self.num_batches, self.num_vars).to(self.device)

        # jitted function
        self.compute_grad = torch.jit.script(_compute_grad)

    @torch.no_grad()
    def reset(self):
        self.yk.zero_()
        self.yk1.zero_()
        self.xk.zero_()
        self.xk1.zero_()
        self.grad.zero_()

        self.Pk.zero_()
        self.ydiff.zero_()
        
    @torch.no_grad()
    def set_data(self, Q, q, A, b, rho, lb, ub):
        self.Q = Q.to(self.device)
        self.q = q.to(self.device)
        
        self.A = A.to(self.device)
        self.b = b.to(self.device)
        self.rho = rho
        
        self.lb = lb.to(self.device)
        self.ub = ub.to(self.device)
        
        self.hessian = self.compute_hessian()
        self.aux2, self.aux3 = self.compute_aux()
        
    @torch.no_grad()
    def update(self):
        def _batchless(P, A, x, b):
            return P + A @ x - b
        
        self.Pk = functorch.vmap(_batchless)(self.Pk, self.A, self.xk, self.b)
        self.aux1, self.aux2 = self.compute_aux()
    
    @torch.no_grad()
    def compute_hessian(self):
        def _batchless(Q, A):
            return 2 * (Q + self.rho * A.T @ A)
        return functorch.vmap(_batchless)(self.Q, self.A)
    
    @torch.no_grad()
    def compute_aux(self):
        def _batchless(Q, q, A, b, P):
            aux1 = -b + P
            aux2 = 2.0 * self.rho * A.T  @ aux1 + q
            return aux1, aux2
        return functorch.vmap(_batchless)(self.Q, self.q, self.A, self.b, self.Pk)
    
    @torch.no_grad()
    def compute_eqc_err(self, xk):
        def _batchless(A, b, x):
            return A @ x - b
        return functorch.vmap(_batchless)(self.A, self.b, xk)
    
    @torch.no_grad()
    def compute_obj_unconstrained(self, xk):
        def _batchless(Q, q, A, b, P, x):
            obj = x @ Q @ x + q @ x
            return obj
        
        return functorch.vmap(_batchless)(self.Q, self.q, self.A, self.b, self.Pk, xk)
    
    @torch.no_grad()
    def compute_obj(self, xk):
        def _batchless(Q, q, A, b, P, x):
            obj = x @ Q @ x + q @ x + self.rho * torch.norm(A @ x + self.aux2)
            return obj
        
        return functorch.vmap(_batchless)(self.Q, self.q, self.A, self.b, self.Pk, xk)
    
    @torch.no_grad()
    def update_grad(self):
        self.grad = self.compute_grad(self.yk, self.hessian, self.aux2)


class ForceQP(object):
    @torch.no_grad()
    def __init__(self, num_batches, num_vars, friction_coeff=None, device=torch.device("cuda:0")):
        self.device = device
        
        # problem params
        self.num_vars = num_vars
        self.num_batches = num_batches

        if isinstance(friction_coeff, torch.Tensor):
            self.friction_coeff = friction_coeff.to(device).view(num_batches, 1)
        elif isinstance(friction_coeff, float):
            self.friction_coeff = friction_coeff * torch.ones(num_batches, 1).to(device)
        else:
            self.friction_coeff = None
        
        # internal states
        self.xk = torch.zeros(self.num_batches, self.num_vars).to(self.device)
        self.yk = torch.zeros(self.num_batches, self.num_vars).to(self.device)
        # # if it's a force problem, initialize the forces as the unit contact normal
        # self.xk[:, 2:3] = 1.0
        # self.yk[:, 2:3] = 1.0
        self.xk1 = torch.zeros(self.num_batches, self.num_vars).to(self.device)
        self.yk1 = torch.zeros(self.num_batches, self.num_vars).to(self.device)
        self.zk = torch.zeros(self.num_batches, self.num_vars).to(self.device)
        self.grad = torch.zeros(self.num_batches, self.num_vars).to(self.device)

        self.ydiff = torch.zeros(self.num_batches, self.num_vars).to(self.device)

        # jitted function
        self.compute_grad = torch.jit.script(_compute_grad)

    @torch.no_grad()
    def reset(self):
        self.yk.zero_()
        self.yk1.zero_()
        self.xk.zero_()
        self.xk1.zero_()
        self.grad.zero_()
        self.ydiff.zero_()
        
    @torch.no_grad()
    def set_data(self, Q, q, lb, ub):
        self.Q = Q.to(self.device)
        self.q = q.to(self.device)
        
        self.lb = lb.to(self.device)
        self.ub = ub.to(self.device)
        
        self.hessian = 2 * self.Q
        
    @torch.no_grad()
    def update(self):
        pass
    
    @torch.no_grad()
    def compute_obj_unconstrained(self, xk):
        def _batchless(Q, q, x):
            obj = x @ Q @ x + q @ x
            return obj
        
        return functorch.vmap(_batchless)(self.Q, self.q, xk)
    
    @torch.no_grad()
    def compute_obj(self, xk):
        def _batchless(Q, q, x):
            obj = x @ Q @ x + q @ x
            return obj
        
        return functorch.vmap(_batchless)(self.Q, self.q, xk)
    
    @torch.no_grad()
    def update_grad(self):
        self.grad = self.compute_grad(self.yk, self.hessian, self.q)

class FISTA(torch.nn.Module):
    @torch.no_grad()
    def __init__(self, prob, step_size=None, device=torch.device("cuda:0")):
        self.prob = prob
        self.friction_coeff = self.prob.friction_coeff
        self.device = device

        self.tk = 1.0
        self.k = 0
        
        if step_size is not None:
            self.step_size = step_size * torch.ones(prob.num_batches).to(device)
        else:
            self.step_size = self.compute_step_size().to(device)

        # jitted functions
        self.grad_descent = torch.jit.script(_grad_descent)
        self.accelerated_update = torch.jit.script(_accelerated_update)
        self.proj_bounds = torch.jit.script(_proj_bounds)
        self.proj_friction_cone = torch.jit.script(_proj_friction_cone)

    def reset(self):
        self.prob.reset()
        self.tk = 1.0
        self.k = 0
    
    @torch.no_grad()     
    def compute_step_size(self):
        if not torch.cuda.is_available():
            # only cpu and cuda implemented svd; svd on mps not supported at the moment
            hessian = self.prob.hessian.to("cpu")
        else:
            hessian = self.prob.hessian

        _, S, _ = torch.linalg.svd(hessian)
        L, _ = torch.max(S, dim=1)
        step_size = (1 / L).to(self.device)
        
        return step_size.view(self.prob.num_batches, 1)
        
    @torch.no_grad()
    def step(self):
        self.prob.update_grad()

        self.prob.zk = self.grad_descent(self.prob.yk, self.step_size, self.prob.grad)

        # TODO: implement the projection onto the intersecton of the two constraints
        # for now ignore the box constraints if there are friction cone constraints
        if self.friction_coeff is not None:
            self.prob.xk1 = self.proj_friction_cone(self.prob.zk, self.friction_coeff)
        else:
            self.prob.xk1 = self.proj_bounds(self.prob.zk, self.prob.lb, self.prob.ub)

        self.prob.yk, self.tk = self.accelerated_update(self.prob.xk, self.prob.xk1, self.tk)
        self.prob.xk = self.prob.xk1
        self.prob.update()
        
        self.k += 1

def _compute_grad(y, hessian, bias):
    return torch.einsum('bij, bj -> bi', hessian, y) + bias

def _grad_descent(y: torch.Tensor, step_size: torch.Tensor, grad: torch.Tensor):
    return y - step_size * grad

def _proj_bounds(z: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor):
    return torch.min(torch.max(z, lb), ub)

def _accelerated_update(xk: torch.Tensor, xk1: torch.Tensor, tk: float):
    tk1 = 1.0 + (1.0 + 4.0 * tk ** 2) ** 0.5 / 2.0
    return xk1 + ((tk - 1) / tk1) * (xk1 - xk), tk1

def _proj_friction_cone(forces: torch.Tensor, friction_coeff: torch.Tensor):
    '''
    this function projects a batch of 3D forces to their friction cone
    all forces are assumed to be in their local contact frame
    '''
    num_batches, num_vars = forces.shape

    # reshape the forces so that each row has only one 3D force
    b = int(num_batches * num_vars / 3)
    reshaped_forces = forces.view(b, 3)
    mu = friction_coeff.repeat_interleave(int(num_vars / 3), dim=0)
    projected_forces = torch.zeros_like(reshaped_forces)
    
    ft = reshaped_forces[:, :2] # tangential force
    fn = reshaped_forces[:, 2:] # normal force
    norm_ft = torch.norm(ft, dim=1, keepdim=True)
    
    # if the forces are not unilateral, set them to zeros
    not_unilateral = fn <= 0 # hard-coded threshold
    reshaped_forces[not_unilateral.view(-1)] = 0
    
    # if the forces are outside the friction cone, project them onto the cone    
    beta = torch.zeros_like(reshaped_forces)
    gamma = torch.zeros_like(reshaped_forces)
    numerator = (mu ** 2) * norm_ft + (mu * fn)
    denominator = ((mu ** 2) + 1) * norm_ft    
    beta[:, :2] = (numerator / denominator).repeat(1, 2)
    gamma[:, 2:] = (mu * norm_ft + fn) / ((mu ** 2) + 1.0)

    # for forces that are either inside the cone or not unilateral, no change needs to be made
    inside_cone = norm_ft <= mu * fn
    no_change = inside_cone + not_unilateral
    
    beta[no_change.view(-1)] = 1.
    gamma[no_change.view(-1)] = 0.
    projected_forces = beta * reshaped_forces + gamma
    
    return projected_forces.view(num_batches, num_vars)
