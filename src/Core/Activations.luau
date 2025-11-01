return {
	relu = function(v)
		for i = 1, #v do
			v[i] = v[i] < 0 and 0 or v[i]
		end
		return v
	end,

	softmax = function(v)
		local maxVal = -math.huge
		for i = 1, #v do
			maxVal = v[i] > maxVal and v[i] or maxVal
		end

		local sum = 0
		for i = 1, #v do
			v[i] = math.exp(v[i] - maxVal)
			sum += v[i]
		end

		for i = 1, #v do
			v[i] /= sum
		end

		return v
	end
}
