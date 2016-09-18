local KLCriterion, parent = torch.class('nn.KLCriterion', 'nn.Criterion')


function KLCriterion:__init(scale)

    self.scale = scale or 1.0
end

function KLCriterion:updateOutput(input, target)

    local m1 = input[1]
    local var1 = input[2]

    local m2 = target[1] or torch.Tensor():resizeAs(m1):zero()
    local var2 = target[2] or torch.Tensor():resizeAs(var1):fill(1)

    -- KL  = 1/2log(var2/var1) + 1/(2*var2) * (var1 + (mu1 - mu2 )^2) - 1/2
    -- KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
    -- KL(m2 = 0 ,var2 = 1) =  - 1/2log(var1) + 1/2(var1 + m1^2) - 1/2 

    local KLDelements = m1 - m2
    KLDelements:pow(2):add(var1):cdiv(var2 + 1e-10)
               :add(-1)
               :add(-1, torch.log(var1)):add(torch.log(var2))
               :mul(0.5)

    self.output = torch.sum(KLDelements)
 
    return self.output
end

function KLCriterion:updateGradInput(input, target)
    self.gradInput = {}


    local m1 = input[1]
    local var1 = input[2]
    local m2 = target[1] or torch.Tensor():resizeAs(m1):zero()
    local var2 = target[2] or torch.Tensor():resizeAs(var1):fill(1)

    -- dKL_dm1 = (m1 - m2)/var2
    self.gradInput[1] = m1:clone():add(-1, m2):cdiv(var2 + 1e-10)

    -- dKL_dvar1 = 0.5 * (1/var2 - 1/var1)
    self.gradInput[2] = self.gradInput[2] or var1.new()
    self.gradInput[2]:resizeAs(var1)

    self.gradInput[2]:copy(var2):pow(-1):add(-1, torch.pow(var1, -1) ):mul(0.5)

    self.gradInput[1]:mul(self.scale)
    self.gradInput[2]:mul(self.scale)

    return self.gradInput
end

