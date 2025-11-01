local Constants = require(script.Parent.Constants)

local CANVAS_SIZE = Constants.CANVAS_SIZE
local NOISE_THRESHOLD = 0.01

-- Purpose: Finds the bounding box of the drawn content
-- Arguments: canvasData (2D array of pixel values [0,1])
-- Returns: hasContent (bool), minX, maxX, minY, maxY
-- Time Complexity: O(n²)
local function findBoundingBox(canvasData)
	local minX, maxX = CANVAS_SIZE, 1
	local minY, maxY = CANVAS_SIZE, 1
	local hasContent = false
	
	for y = 1, CANVAS_SIZE do
		for x = 1, CANVAS_SIZE do
			if canvasData[y][x] > NOISE_THRESHOLD then
				hasContent = true
				minX = math.min(minX, x)
				maxX = math.max(maxX, x)
				minY = math.min(minY, y)
				maxY = math.max(maxY, y)
			end
		end
	end
	
	return hasContent, minX, maxX, minY, maxY
end

-- Purpose: Calculates the center of mass of the drawn content
-- Arguments: canvasData (2D array), minX, maxX, minY, maxY (bounding box)
-- Returns: centerX, centerY
-- Time Complexity: O(n²)
local function calculateCenterOfMass(canvasData, minX, maxX, minY, maxY)
	local totalMass = 0
	local centerX = 0
	local centerY = 0
	
	for y = minY, maxY do
		for x = minX, maxX do
			local val = canvasData[y][x]
			totalMass = totalMass + val
			centerX = centerX + x * val
			centerY = centerY + y * val
		end
	end
	
	if totalMass > 0 then
		centerX = centerX / totalMass
		centerY = centerY / totalMass
	else
		centerX = (minX + maxX) / 2
		centerY = (minY + maxY) / 2
	end
	
	return centerX, centerY
end

-- Purpose: Applies bilinear interpolation shift to center the content
-- Arguments: canvasData (2D array), shiftX, shiftY (centering offset)
-- Returns: centered (2D array)
-- Time Complexity: O(n²)
local function applyCenteringShift(canvasData, shiftX, shiftY)
	local centered = {}
	for y = 1, CANVAS_SIZE do
		centered[y] = {}
		for x = 1, CANVAS_SIZE do
			centered[y][x] = 0
		end
	end
	
	for y = 1, CANVAS_SIZE do
		for x = 1, CANVAS_SIZE do
			if canvasData[y][x] > 0 then
				local newX = x + shiftX
				local newY = y + shiftY
				local x1 = math.floor(newX)
				local y1 = math.floor(newY)
				local x2 = x1 + 1
				local y2 = y1 + 1
				
				local fx = newX - x1
				local fy = newY - y1
				
				local value = canvasData[y][x]
				
				if x1 >= 1 and x1 <= CANVAS_SIZE and y1 >= 1 and y1 <= CANVAS_SIZE then
					centered[y1][x1] = centered[y1][x1] + value * (1 - fx) * (1 - fy)
				end
				if x2 >= 1 and x2 <= CANVAS_SIZE and y1 >= 1 and y1 <= CANVAS_SIZE then
					centered[y1][x2] = centered[y1][x2] + value * fx * (1 - fy)
				end
				if x1 >= 1 and x1 <= CANVAS_SIZE and y2 >= 1 and y2 <= CANVAS_SIZE then
					centered[y2][x1] = centered[y2][x1] + value * (1 - fx) * fy
				end
				if x2 >= 1 and x2 <= CANVAS_SIZE and y2 >= 1 and y2 <= CANVAS_SIZE then
					centered[y2][x2] = centered[y2][x2] + value * fx * fy
				end
			end
		end
	end
	
	return centered
end

-- Purpose: Flattens 2D canvas to 1D array for model input
-- Arguments: canvasData (2D array)
-- Returns: arr (1D array)
-- Time Complexity: O(n²)
local function flattenCanvas(canvasData)
	local arr = {}
	for y = 1, CANVAS_SIZE do
		for x = 1, CANVAS_SIZE do
			local value = math.min(1, canvasData[y][x])
			table.insert(arr, value)
		end
	end
	return arr
end

-- Purpose: Creates empty 1D array for empty canvas
-- Returns: arr (1D array of zeros)
-- Time Complexity: O(n²)
local function createEmptyArray()
	local arr = {}
	for i = 1, CANVAS_SIZE * CANVAS_SIZE do
		table.insert(arr, 0)
	end
	return arr
end

-- Purpose: Preprocesses canvas data for MNIST model input
-- Centers the digit and applies proper scaling
-- Arguments: canvasData (2D array of pixel values [0,1])
-- Returns: arr (1D array ready for model)
-- Time Complexity: O(n²)
local function preprocessCanvas(canvasData)
	local hasContent, minX, maxX, minY, maxY = findBoundingBox(canvasData)
	
	if not hasContent then
		return createEmptyArray()
	end
	
	local centerX, centerY = calculateCenterOfMass(canvasData, minX, maxX, minY, maxY)
	
	local targetX = CANVAS_SIZE / 2 + 0.5
	local targetY = CANVAS_SIZE / 2 + 0.5
	
	local shiftX = targetX - centerX
	local shiftY = targetY - centerY
	
	local centered = applyCenteringShift(canvasData, shiftX, shiftY)
	
	return flattenCanvas(centered)
end

return {
	preprocessCanvas = preprocessCanvas
}