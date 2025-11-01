local UserInputService = game:GetService("UserInputService")
local RunService = game:GetService("RunService")

local Model = require(game.ReplicatedStorage.Core.Model)

local CANVAS_SIZE = 28
local PIXEL_SIZE = 12
local CANVAS_PADDING = 8
local BRUSH_SIZE = 2

local COLORS = {
	bg = Color3.fromRGB(20, 20, 20),
	surface = Color3.fromRGB(30, 30, 30),
	text = Color3.fromRGB(220, 220, 220),
	textMuted = Color3.fromRGB(120, 120, 120),
	danger = Color3.fromRGB(180, 80, 80),
}

local screenGui = Instance.new("ScreenGui")
screenGui.Name = "MNISTGui"
screenGui.ResetOnSpawn = false
screenGui.Parent = game.Players.LocalPlayer:WaitForChild("PlayerGui")

local mainContainer = Instance.new("Frame")
mainContainer.Name = "MainContainer"
mainContainer.Size = UDim2.new(0, 420, 0, 500)
mainContainer.Position = UDim2.new(0.5, -210, 0.5, -250)
mainContainer.BackgroundColor3 = COLORS.surface
mainContainer.BorderSizePixel = 0
mainContainer.Parent = screenGui

local topPredLabel = Instance.new("TextLabel")
topPredLabel.Name = "TopPredLabel"
topPredLabel.Size = UDim2.new(1, -20, 0, 40)
topPredLabel.Position = UDim2.new(0, 10, 0, 10)
topPredLabel.BackgroundTransparency = 1
topPredLabel.TextColor3 = COLORS.text
topPredLabel.TextSize = 32
topPredLabel.Font = Enum.Font.GothamBold
topPredLabel.Text = "? (0%)"
topPredLabel.TextXAlignment = Enum.TextXAlignment.Center
topPredLabel.Parent = mainContainer

local canvasFrame = Instance.new("Frame")
canvasFrame.Name = "Canvas"
canvasFrame.Size = UDim2.new(0, CANVAS_SIZE * PIXEL_SIZE + CANVAS_PADDING * 2, 0, CANVAS_SIZE * PIXEL_SIZE + CANVAS_PADDING * 2)
canvasFrame.Position = UDim2.new(0.5, -(CANVAS_SIZE * PIXEL_SIZE + CANVAS_PADDING * 2) / 2, 0, 60)
canvasFrame.BackgroundColor3 = Color3.fromRGB(10, 10, 15)
canvasFrame.BorderSizePixel = 0
canvasFrame.Parent = mainContainer

-- Pixel grid (28x28)
local pixels = {}
for y = 1, CANVAS_SIZE do
	pixels[y] = {}
	for x = 1, CANVAS_SIZE do
		local pixel = Instance.new("Frame")
		pixel.Name = "Pixel_" .. x .. "_" .. y
		pixel.Size = UDim2.new(0, PIXEL_SIZE, 0, PIXEL_SIZE)
		pixel.Position = UDim2.new(0, CANVAS_PADDING + (x - 1) * PIXEL_SIZE, 0, CANVAS_PADDING + (y - 1) * PIXEL_SIZE)
		pixel.BackgroundColor3 = Color3.fromRGB(0, 0, 0)
		pixel.BorderSizePixel = 0
		pixel.Parent = canvasFrame
		pixels[y][x] = { frame = pixel, value = 0 }
	end
end

-- Prediction grid (5x2 simple text labels)
local predictionBars = {}
local gridLayout = Instance.new("Frame")
gridLayout.Name = "GridLayout"
gridLayout.Size = UDim2.new(1, -20, 0, 80)
gridLayout.Position = UDim2.new(0, 10, 0, 360)
gridLayout.BackgroundTransparency = 1
gridLayout.Parent = mainContainer

for digit = 0, 9 do
	local col = digit % 5
	local row = math.floor(digit / 5)
	
	local cellContainer = Instance.new("Frame")
	cellContainer.Name = "Cell_" .. digit
	cellContainer.Size = UDim2.new(0.2, -2, 0.5, -2)
	cellContainer.Position = UDim2.new(col * 0.2, col * 2, row * 0.5, row * 2)
	cellContainer.BackgroundColor3 = Color3.fromRGB(50, 50, 50)
	cellContainer.BorderSizePixel = 0
	cellContainer.Parent = gridLayout
	
	local digitNum = Instance.new("TextLabel")
	digitNum.Name = "Number"
	digitNum.Size = UDim2.new(1, 0, 0, 14)
	digitNum.Position = UDim2.new(0, 0, 0, 2)
	digitNum.BackgroundTransparency = 1
	digitNum.TextColor3 = COLORS.textMuted
	digitNum.TextSize = 10
	digitNum.Font = Enum.Font.Gotham
	digitNum.Text = tostring(digit)
	digitNum.Parent = cellContainer
	
	local percentLabel = Instance.new("TextLabel")
	percentLabel.Name = "Percent"
	percentLabel.Size = UDim2.new(1, 0, 1, -16)
	percentLabel.Position = UDim2.new(0, 0, 0, 14)
	percentLabel.BackgroundTransparency = 1
	percentLabel.TextColor3 = COLORS.text
	percentLabel.TextSize = 14
	percentLabel.Font = Enum.Font.GothamBold
	percentLabel.Text = "0%"
	percentLabel.Parent = cellContainer
	
	predictionBars[digit] = { label = percentLabel }
end

local clearButton = Instance.new("TextButton")
clearButton.Name = "ClearButton"
clearButton.Size = UDim2.new(0, 80, 0, 28)
clearButton.Position = UDim2.new(0.5, -40, 1, -40)
clearButton.BackgroundColor3 = COLORS.danger
clearButton.TextColor3 = COLORS.text
clearButton.TextSize = 12
clearButton.Font = Enum.Font.GothamBold
clearButton.Text = "Clear"
clearButton.BorderSizePixel = 0
clearButton.Parent = mainContainer

local isDrawing = false
local canvasData = {}

local function initializeCanvas()
	for y = 1, CANVAS_SIZE do
		canvasData[y] = {}
		for x = 1, CANVAS_SIZE do
			canvasData[y][x] = 0
		end
	end
end

local function normalizePixel(value)
	return math.min(1, value / 255)
end

local function setPixel(x, y, value)
	if x < 1 or x > CANVAS_SIZE or y < 1 or y > CANVAS_SIZE then
		return
	end
	value = math.min(1, value)
	canvasData[y][x] = math.max(canvasData[y][x], value)
	local displayValue = canvasData[y][x] * 255
	pixels[y][x].frame.BackgroundColor3 = Color3.fromRGB(displayValue, displayValue, displayValue)
end

local function drawBrush(centerX, centerY, intensity)
	for dy = -BRUSH_SIZE, BRUSH_SIZE do
		for dx = -BRUSH_SIZE, BRUSH_SIZE do
			local dist = math.sqrt(dx * dx + dy * dy)
			if dist <= BRUSH_SIZE then
				local falloff = 1 - (dist / BRUSH_SIZE)
				setPixel(centerX + dx, centerY + dy, intensity * falloff)
			end
		end
	end
end

-- Get canvas as flat array
local function getCanvasArray()
	local arr = {}
	for y = 1, CANVAS_SIZE do
		for x = 1, CANVAS_SIZE do
			table.insert(arr, canvasData[y][x])
		end
	end
	return arr
end

local function updatePrediction()
	local input = getCanvasArray()
	local output = Model.predict(input)
	
	-- Find top prediction
	local topDigit = 0
	local topConfidence = 0
	for digit = 0, 9 do
		local confidence = output[digit + 1] or 0
		if confidence > topConfidence then
			topConfidence = confidence
			topDigit = digit
		end
	end
	
	-- Update top label
	local topPercent = math.floor(topConfidence * 100 + 0.5)
	topPredLabel.Text = topDigit .. " (" .. topPercent .. "%)"
	
	-- Update grid
	for digit = 0, 9 do
		local confidence = output[digit + 1] or 0
		local percent = math.floor(confidence * 100 + 0.5)
		predictionBars[digit].label.Text = percent .. "%"
	end
end

-- Clear canvas
local function clearCanvas()
	initializeCanvas()
	for y = 1, CANVAS_SIZE do
		for x = 1, CANVAS_SIZE do
			pixels[y][x].frame.BackgroundColor3 = Color3.fromRGB(0, 0, 0)
		end
	end
	updatePrediction()
end

-- Input handling
UserInputService.InputBegan:Connect(function(input, gameProcessed)
	if gameProcessed then return end
	if input.UserInputType == Enum.UserInputType.MouseButton1 then
		isDrawing = true
	end
end)

UserInputService.InputEnded:Connect(function(input, gameProcessed)
	if input.UserInputType == Enum.UserInputType.MouseButton1 then
		isDrawing = false
	end
end)

-- Drawing loop
RunService.RenderStepped:Connect(function()
	if isDrawing then
		local mouse = game.Players.LocalPlayer:GetMouse()
		local canvasPos = canvasFrame.AbsolutePosition
		local canvasSize = canvasFrame.AbsoluteSize
		
		local mouseX = mouse.X - canvasPos.X
		local mouseY = mouse.Y - canvasPos.Y
		
		if mouseX >= 0 and mouseX < canvasSize.X and mouseY >= 0 and mouseY < canvasSize.Y then

			local gridX = math.floor((mouseX - CANVAS_PADDING) / PIXEL_SIZE) + 1
			local gridY = math.floor((mouseY - CANVAS_PADDING) / PIXEL_SIZE) + 1
			
			if gridX >= 1 and gridX <= CANVAS_SIZE and gridY >= 1 and gridY <= CANVAS_SIZE then
				drawBrush(gridX, gridY, 2)
				updatePrediction()
			end
		end
	end
end)

clearButton.MouseButton1Click:Connect(function()
	clearCanvas()
end)

-- Initialize
initializeCanvas()
updatePrediction()