local Constants = require(script.Parent.Constants)

local CANVAS_SIZE = Constants.CANVAS_SIZE
local PIXEL_SIZE = Constants.PIXEL_SIZE
local CANVAS_PADDING = Constants.CANVAS_PADDING
local COLORS = Constants.COLORS

-- Purpose: Creates and returns all UI elements for the MNIST interface
-- Returns: Table with UI references (pixels, predictionBars, clearButton, topPredLabel, canvasFrame)
-- Time Complexity: O(n²) where n = CANVAS_SIZE
return {
	createUI = function()

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
	
	return {
		pixels = pixels,
		predictionBars = predictionBars,
		clearButton = clearButton,
		topPredLabel = topPredLabel,
		canvasFrame = canvasFrame
	}
end
}