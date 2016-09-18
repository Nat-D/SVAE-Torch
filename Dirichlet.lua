local Dirichlet, parent = torch.class( 'nn.Dirichlet', 'nn.Module' )
require 'cephes'

function Dirichlet:__init(K)
	parent.__init(self)
	self.K = K
	self.a = torch.Tensor(K)
	self.Ga = torch.Tensor(K)
	self.a0 = torch.Tensor(K)

	self.stats = torch.Tensor(K)
end

function Dirichlet:parameters()
	return {self.a}, {self.Ga}
end


function Dirichlet:setPrior(a0)
	self.a0:fill(a0)
end

function Dirichlet:setParameters(a)
	self.a:copy(a) 
end

function Dirichlet:updateExpectedStats()
	-- Sufficient statistics t() = [logP(k=1), logP(k=2), .. ]
	-- <logP(k)> = digamma(a[k]) - digamma( sum(a) )
	local a = self.a:clone()
	self.stats:zero()
	self.stats:add(cephes.digamma(a)):add(-cephes.digamma(a:sum()))

	return self.stats
end


function Dirichlet:accGradParameters(input, gradOutput, scale)
	-- Accumulate Natural Gradient 
	self.Ga:add(self.a):add(-1, self.a0):add(-1, input[3][{{},1}])
	
end














