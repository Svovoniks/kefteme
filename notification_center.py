import cv2
import telebot
import time
import io
class Notifier:
	FALL_DOWN_MESSAGE_TEMPLATE = '''/////////////////////
Fall detected on camera {}
/////////////////////
'''
	def __init__(self, camera_name, config):
		self.config = config
		self.camera_name = camera_name
		self.bot = telebot.TeleBot(self.config['BOT']['token'])
		self.chat_id = int(self.config['BOT']['chat_id'])
		self.mute_duration = int(self.config['BOT']['mute_duration'])
		self.last_notification_time = 0

	def check_time(self):
		return time.time() - self.last_notification_time > self.mute_duration

	def handle_action(self, action, frame):
		if action == 'Fall Down' and self.check_time():
			try:
				s, buffer = cv2.imencode('.jpg', frame)
				image = io.BytesIO(buffer)
				if not s:
					raise Exception('Failed to get image')
			except Exception as e:
				image = None
				print(e)

			self.notify(Notifier.FALL_DOWN_MESSAGE_TEMPLATE.format(self.camera_name), image)

	def notify(self, message, image):
		if image != None:
			self.bot.send_photo(self.chat_id, image, caption=message)
		else:
			self.bot.send_message(self.chat_id, message)

		self.last_notification_time = time.time()


