import configparser
import telebot

class Config:
	CONFIG_PATH = '.cfg'

	def __init__(self):
		self.config = configparser.ConfigParser()
		self.config.read(Config.CONFIG_PATH)

	def update(self, new_config):
		with open(Config.CONFIG_PATH, 'w') as file:
			new = self.config.write(file)

	def get_chat_id(self):
		bot = telebot.TeleBot(self.config['BOT']['token'])

		updates = bot.get_updates()
		if len(updates) == 0:
			raise Exception("couldn't get chat_id from the bot, try to add it to a group or texting him")

		return updates[-1].message.chat.id

	def validate_dict(self, og_item, item):
		if type(og_item) == type({}):
			for i in og_item.keys():
				if i not in item:
					raise Exception(f'missing config for {i}')
				self.validate_dict(og_item[i], item[i])


	def validate(self):
		template = {	
			'BOT': {'token': '<TOKEN>', 'mute_duration' : 'mute_duration'},
			'GENERAL': {'device': 'cpu', 'grid_width' : 'width', 'video_height': 'height'},
			'SOURCES' : 'sources'
		}
		self.validate_dict(template, self.config)

		if 'chat_id' not in self.config['BOT']:
			chat_id = self.get_chat_id()
			self.config['BOT']['chat_id'] = str(self.get_chat_id())
			self.update(self.config)
			

