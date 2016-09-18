local NormalGamma, parent = torch.class( 'nn.NormalGamma', 'nn.Module' )

function NormalGamma:__init(K, D)
	parent.__init(self)

	self.K = K
	self.D = D 

	self.weight = torch.Tensor(4, K, D)
	self.gradWeight = torch.Tensor(4, K, D)

	self.prior = torch.Tensor(4, K, D)

	self.stats = torch.Tensor(4, K, D)

end

function NormalGamma:parameters()
	return {self.weight}, {self.gradWeight}
end

function NormalGamma:setPrior(m0, l0, a0, b0)
	--[[
	Expect same prior across cluster K
	m0 = torch.Tensor(D)
	l0 = torch.Tensor(D)
	a0 = torch.Tensor(D)
	b0 = torch.Tensor(D)
	]]--
	local K = self.K
	local D = self.D
	
	self.prior[1]:copy( torch.cmul( m0, l0):repeatTensor(K,1) )
	self.prior[2]:copy( torch.cmul( l0, m0):cmul(m0):add(2,b0):repeatTensor(K,1) )--2*b0 + l*m0*m0
	self.prior[3]:copy( l0:repeatTensor(K,1) )
	self.prior[4]:copy( (2*a0 - 1):repeatTensor(K,1) )

	return self.prior
end

function NormalGamma:setParameters(m1, l1, a1, b1)
	local K = self.K
	local D = self.D

	self.weight[1]:copy( torch.cmul( m1, l1) )
	self.weight[2]:copy( torch.cmul( l1, m1):cmul(m1):add(2,b1) )--2*b0 + l*m0*m0
	self.weight[3]:copy( l1 )
	self.weight[4]:copy( (2*a1 - 1) )

	return self.weight
end

function NormalGamma:accGradParameters(input, gradOutput, scale)
	-- Accumulate Natural gradient
	self.gradWeight:add(self.weight):add(-1, self.prior):add(-1, input)
	
	if gradOutput then
		self.gradWeight[1]:add(-1, gradOutput[1])
		self.gradWeight[2]:add(-1, gradOutput[2])
	end
end

function NormalGamma:getBasicParameters()
	-- w1 = ml
	-- w2 = 2b + lm^2
	-- w3 = l
	-- w4 = 2a -1
	local l = self.weight[3] --[KxD]
	local m = torch.cdiv( self.weight[1], l )
	local a = 0.5 * (self.weight[4] + 1)
	local b = 0.5 * (self.weight[2] - torch.cmul( self.weight[1], m) )

	return {m, l, a, b}
end

require 'cephes'
function NormalGamma:updateExpectedStats()
	--[[
	t(gamma, mu) = [ g * mu,
					 -0.5 g,
					 -0.5 g *mu*mu,
					 0.5 * log(g)
					]
	<t(g,m)> = [ (a/b) * m,
				 -0.5 (a/b),
				 -0.5 (1/l + m*m*a/b),
				 0.5 * ( digamma(a) - ln(b) )
				 ]
	]]--
	local K = self.K
	local D = self.D
	local m, l, a, b = unpack( self:getBasicParameters() )
	self.stats[1]:copy(m):cmul(a):cdiv(b + 1e-10)
	self.stats[2]:copy(a):cdiv(b + 1e-10):mul(-0.5)
	self.stats[3]:copy(m):cmul(m):cmul(a):cdiv(b + 1e-10):add(torch.pow(l,-1)):mul(-0.5)

	local a_flat = a:view(K*D)
	a_flat:copy( cephes.digamma(a_flat) )
	self.stats[4]:copy(a):add(-1, torch.log(b)):mul(0.5)

	return self.stats
end
	

















