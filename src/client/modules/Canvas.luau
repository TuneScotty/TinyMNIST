local Constants = require(script.Parent.Constants)

local CANVAS_SIZE = Constants.CANVAS_SIZE
local BRUSH_SIZE = Constants.BRUSH_SIZE

-- Purpose: Manages canvas data state and drawing operations with sub-pixel precision
-- Arguments: pixels (UI pixel grid reference)
-- Time Complexity: O(1) for pixel operations, O(BRUSH_SIZE²) for brush drawing
local function createCanvas(pixels)
	local canvasData = {}
	
	for y = 1, CANVAS_SIZE do
		canvasData[y] = {}
		for x = 1, CANVAS_SIZE do
			canvasData[y][x] = 0
		end
	end
	
	local function setPixel(y, x, value)
		if x < 1 or x > CANVAS_SIZE or y < 1 or y > CANVAS_SIZE then
			return
		end
		canvasData[y][x] = math.min(1, canvasData[y][x] + value)
		local displayValue = canvasData[y][x] * 255
		pixels[y][x].frame.BackgroundColor3 = Color3.fromRGB(displayValue, displayValue, displayValue)
	end
	
	local function setPixelSmooth(y, x, value)
		local x1 = math.floor(x)
		local y1 = math.floor(y)
		local x2 = x1 + 1
		local y2 = y1 + 1
		
		local fx = x - x1
		local fy = y - y1
		
		setPixel(y1, x1, value * (1 - fx) * (1 - fy))
		setPixel(y1, x2, value * fx * (1 - fy))
		setPixel(y2, x1, value * (1 - fx) * fy)
		setPixel(y2, x2, value * fx * fy)
	end
	
	local function drawBrush(centerY, centerX, intensity)
		for dy = -BRUSH_SIZE, BRUSH_SIZE do
			for dx = -BRUSH_SIZE, BRUSH_SIZE do
				local dist = math.sqrt(dx * dx + dy * dy)
				if dist <= BRUSH_SIZE then
					local sigma = BRUSH_SIZE / 2
					local falloff = math.exp(-(dist * dist) / (2 * sigma * sigma))
					setPixelSmooth(centerY + dy, centerX + dx, intensity * falloff)
				end
			end
		end
	end
	
	local function clear()
		for y = 1, CANVAS_SIZE do
			for x = 1, CANVAS_SIZE do
				canvasData[y][x] = 0
				pixels[y][x].frame.BackgroundColor3 = Color3.fromRGB(0, 0, 0)
			end
		end
	end
	
	local function getData()
		return canvasData
	end
	
	return {
		drawBrush = drawBrush,
		clear = clear,
		getData = getData
	}
end

return {
	createCanvas = createCanvas
}