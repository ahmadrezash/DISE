from torchvision.transforms import Compose, Grayscale, Resize, ToPILImage


class InputDataTransform(Compose):

	def __init__(self):
		transforms = [
			Resize((200, 200)),
			# Grayscale()
		]
		super().__init__(transforms)


class OutputDataTransform(Compose):

	def __init__(self):
		transforms = [
			ToPILImage()
		]
		super().__init__(transforms)
