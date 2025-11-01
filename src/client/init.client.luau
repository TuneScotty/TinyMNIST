local UserInputService = game:GetService("UserInputService")
local RunService = game:GetService("RunService")

local Model = require(game.ReplicatedStorage.Core.Model)
local Constants = require(script.modules.Constants)
local UI = require(script.modules.UI)
local CanvasModule = require(script.modules.Canvas)
local Preprocessing = require(script.modules.Preprocessing)

local CANVAS_SIZE = Constants.CANVAS_SIZE
local PIXEL_SIZE = Constants.PIXEL_SIZE
local CANVAS_PADDING = Constants.CANVAS_PADDING
local PREDICTION_DEBOUNCE = Constants.PREDICTION_DEBOUNCE

local ui = UI.createUI()
local pixels = ui.pixels
local predictionBars = ui.predictionBars
local clearButton = ui.clearButton
local topPredLabel = ui.topPredLabel
local canvasFrame = ui.canvasFrame

local canvas = CanvasModule.createCanvas(pixels)

local isDrawing = false
local lastMousePos = nil
local lastPredictionTime = 0

local function updatePrediction()
	local canvasData = canvas.getData()
	local input = Preprocessing.preprocessCanvas(canvasData)
	local output = Model.predict(input)
	
	local topDigit = 0
	local topConfidence = 0
	for digit = 0, 9 do
		local confidence = output[digit + 1] or 0
		if confidence > topConfidence then
			topConfidence = confidence
			topDigit = digit
		end
	end
	
	local topPercent = math.floor(topConfidence * 100 + 0.5)
	topPredLabel.Text = topDigit .. " (" .. topPercent .. "%)"
	
	for digit = 0, 9 do
		local confidence = output[digit + 1] or 0
		local percent = math.floor(confidence * 100 + 0.5)
		predictionBars[digit].label.Text = percent .. "%"
	end
end

local function clearCanvas()
	canvas.clear()
	updatePrediction()
end

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

RunService.RenderStepped:Connect(function(dt)
	if isDrawing then
		local mouse = game.Players.LocalPlayer:GetMouse()
		local canvasPos = canvasFrame.AbsolutePosition
		local canvasSize = canvasFrame.AbsoluteSize
		
		local mouseX = mouse.X - canvasPos.X
		local mouseY = mouse.Y - canvasPos.Y
		
		if mouseX >= 0 and mouseX < canvasSize.X and mouseY >= 0 and mouseY < canvasSize.Y then
			local gridX = (mouseX - CANVAS_PADDING) / PIXEL_SIZE + 0.5
			local gridY = (mouseY - CANVAS_PADDING) / PIXEL_SIZE + 0.5
			
			if gridX >= 0.5 and gridX <= CANVAS_SIZE + 0.5 and gridY >= 0.5 and gridY <= CANVAS_SIZE + 0.5 then
				local intensity = 0.8
				
				if lastMousePos then
					local dx = gridX - lastMousePos.X
					local dy = gridY - lastMousePos.Y
					local velocity = math.sqrt(dx * dx + dy * dy)
					intensity = math.max(0.6, math.min(1.0, 1.0 / (1 + velocity)))
				end
				
				lastMousePos = { X = gridX, Y = gridY }
				canvas.drawBrush(gridY, gridX, intensity)
				
				local currentTime = tick()
				if currentTime - lastPredictionTime >= PREDICTION_DEBOUNCE then
					updatePrediction()
					lastPredictionTime = currentTime
				end
			end
		end
	else
		lastMousePos = nil
	end
end)

clearButton.MouseButton1Click:Connect(function()
	clearCanvas()
end)

updatePrediction()