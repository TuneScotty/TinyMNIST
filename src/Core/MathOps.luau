return {
	dot = function(W, x)
		local y = table.create(#W)
		for i = 1, #W do
			local sum = 0
			local row = W[i]
			for j = 1, #row do
				sum += row[j] * x[j]
			end
			y[i] = sum
		end
		return y
	end,

	add_bias = function(v, b)
		for i = 1, #v do
			v[i] += b[i]
		end
		return v
	end
}
