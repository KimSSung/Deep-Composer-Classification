import py_midicsv
import os
import numpy as np
import pandas as pd
import csv
import sys


class Converter:
	def __init__(self, args):
		self.config = args

		self.atype = '' # default

		self.csv_printable = True

		### Set File directory
		# Get the Header and other data at original Midi Data
		self.npy_path = self.config.to_convert_path
		self.origin_midi_dir = "/data/genres/" # Classical/...
		self.output_file_dir = "/data/temp_converter/midi/"
		self.csv_output_dir = "/data/temp_converter/csv/"

	# --------------------------------------------------------------------------
	# functions

	def start_track_string(self, track_num):
		return str(track_num) + ", 0, Start_track\n"

	def title_track_string(self, track_num):
		return str(track_num) + ', 0, Title_t, "Test file"\n'

	def program_c_string(self, track_num, channel, program_num):
		return (
			str(track_num)
			+ ", 0, Program_c, "
			+ str(channel)
			+ ", "
			+ str(int(program_num))
			+ "\n"
		)

	def note_on_event_string(self, track_num, delta_time, channel, pitch, velocity):
		return (
			str(track_num)
			+ ", "
			+ str(delta_time)
			+ ", Note_on_c, "
			+ str(channel)
			+ ", "
			+ str(pitch)
			+ ", "
			+ str(velocity)
			+ "\n"
		)

	def note_off_event_string(self, track_num, delta_time, channel, pitch, velocity):
		return (
			str(track_num)
			+ ", "
			+ str(delta_time)
			+ ", Note_off_c, "
			+ str(channel)
			+ ", "
			+ str(pitch)
			+ ", "
			+ str(velocity)
			+ "\n"
		)

	def end_track_string(self, track_num, delta_time):
		return str(track_num) + ", " + str(delta_time) + ", End_track\n"

	def run(self):

		print(">>>>>> Converting <<<<<<")
		print("PATH: " + self.npy_path + "\n")

		self.convert_file()

		return



	def convert_file(self):

		off_note = 0
		success_num = 0
		new_csv_string = []

		total_track = 0
		track_num = 1  # Set the Track number
		
		program_num = 0
		delta_time = 0
		channel = 0
		pitch = 60
		velocity = 90


		for file in os.listdir(self.npy_path):

			# FOR CHECKING
			if success_num == 5: break

			if os.path.isfile(os.path.join(self.npy_path, file)):

				if "vel" in file:
					self.atype = "vel"
				elif "noise" in file:
					self.atype = "noise"
				else:  # origin input2midi
					self.atype = "origin"

				only_file_name = file.replace(self.atype + "_", "").replace(".npy", "")


				this_genre = '' # genre of this midi
				for genre in self.config.genres:
					if genre in self.config.to_convert_path:
						this_genre = genre
						break
				# if 'to_convert_path' contains genre name -> this_genre = genre name
				# else this_genre = ''
				# print(this_genre)


				# FOR SIMULATION
				# if file != 'scn15_11_format0.mid.npy': continue
				# only_file_name = 'scn15_11_format0.mid'

				# print(only_file_name)


				new_csv_string = []
				load_data = np.load(os.path.join(self.npy_path, file))

				if this_genre == '': # 'to_convert_path' not contain genre name
					origin_file = self.origin_midi_dir + only_file_name
				else: # 'to_convert_path' contains genre name	
					origin_file = self.origin_midi_dir + this_genre + '/' + only_file_name
				print("Original file:", origin_file)

				try:
					origin_file_csv = py_midicsv.midi_to_csv(origin_file)
				except:
					print("MIDI_TO_CSV ERROR !!")
					continue

				else:
					print("current file:", file)
					# for string in origin_file_csv:
					#    if 'Program_c' in string: print(string)
					total_track = 2
					current_used_instrument = [-1, -1]
					# find total track num
					for instrument_num, lst in enumerate(load_data):  # instrument_num : 0-127
						if np.sum(lst) != (off_note) * 400 * 128:
							total_track += 1
							current_used_instrument.append(instrument_num)

					# slower by 4.8
					header = origin_file_csv[0].split(", ")
					# print('Before header:', header)
					header[-1] = str(int(int(header[-1][:-1]) / 4.0)) + "\n"
					header[-2] = str(int(total_track))
					# print('After header:', header)
					new_csv_string.append(
						", ".join(header)
					)  # header_string(total_track) + change last to 168 (too fast)
					new_csv_string.append(origin_file_csv[1])  # self.start_track_string(track_num)

					for string in origin_file_csv:
						if "SMPTE_offset" in string:
							# print(string)
							continue
						elif "Time_signature" in string or "Tempo" in string:
							new_csv_string.append(string)

						elif "Program_c" in string:
							break

					new_csv_string.append(self.end_track_string(track_num, delta_time))
					# print('Before Real Data Part:')
					# for string in new_csv_string: print(string)

					# ## Real Data Part # deleted after add 128 instrument dim
					# current_used_instrument = [-1, -1]
					# for instrument_num in instrument_dict.keys():
					#     current_used_instrument.append(instrument_num)
					# print(lst.shape)

					# print(total_track)

					# Set the track_string_list to identify different instrument time line
					track_string_list = [[] for i in range(0, total_track)]
					track_string_list[0].append(-1)  # To Generate Error -> Header File
					track_string_list[1].append(-1)  # To Generate Error -> Meta File

					note_on_list = [[] for i in range(0, total_track)]
					note_on_list[0].append(-1)
					note_on_list[1].append(-1)

					note_off_list = [[] for i in range(0, total_track)]
					note_off_list[0].append(-1)
					note_off_list[1].append(-1)

					# print(load_data.shape[0], " ", load_data.shape[1], " ", load_data.shape[2])
					for channel_instrument in range(0, load_data.shape[0]):
						for row in range(0, load_data.shape[1]):
							for col in range(0, load_data.shape[2]):

								if load_data[channel_instrument][row][col] == off_note:
									continue
								else:
									# Set the different condition for attacked Midi Files
									# print('music21 instrument:', load_data[row][col]) # 0-59
									# print('py_midicsv instrument:', program_num_map[load_data[row][col]])

									if (
										len(
											track_string_list[
												current_used_instrument.index(
													channel_instrument
												)
											]
										)
										!= 0
									):
										program_num = (
											channel_instrument  # program_num = instrment num
										)
										pitch = col
										channel = 0
										delta_time = 50 * row
										end_delta_time = 50 * (row + 1)
										velocity = int(
											load_data[channel_instrument][row][col]
										)  # TODO: We should consider later
										note_on_list[track_num].append(
											[track_num, delta_time, channel, pitch, velocity]
										)
										note_off_list[track_num].append(
											[
												track_num,
												end_delta_time,
												channel,
												pitch,
												velocity,
											]
										)
									else:
										# Set the track_string_list new track header - program_c event
										track_num = current_used_instrument.index(
											channel_instrument
										)
										if channel_instrument == 128:
											program_num = 1
										else:
											program_num = channel_instrument
										channel = 0
										pitch = col
										delta_time = 50 * row
										end_delta_time = 50 * (row + 1)
										velocity = int(load_data[channel_instrument][row][col])
										track_string_list[track_num].append(
											self.start_track_string(track_num)
										)
										track_string_list[track_num].append(
											self.title_track_string(track_num)
										)
										track_string_list[track_num].append(
											self.program_c_string(track_num, channel, program_num)
										)
										note_on_list[track_num].append(
											[track_num, delta_time, channel, pitch, velocity]
										)
										note_off_list[track_num].append(
											[
												track_num,
												end_delta_time,
												channel,
												pitch,
												velocity,
											]
										)

							for num in range(2, len(note_on_list)):  # num = track num
								for notes in range(0, len(note_on_list[num])):
									track_string_list[num].append(
										self.note_on_event_string(
											note_on_list[num][notes][0],
											note_on_list[num][notes][1],
											note_on_list[num][notes][2],
											note_on_list[num][notes][3],
											note_on_list[num][notes][4],
										)
									)
							for num in range(2, len(note_off_list)):
								for notes in range(0, len(note_off_list[num])):
									track_string_list[num].append(
										self.note_off_event_string(
											note_off_list[num][notes][0],
											note_off_list[num][notes][1],
											note_off_list[num][notes][2],
											note_off_list[num][notes][3],
											note_off_list[num][notes][4],
										)
									)
							note_on_list = [[] for i in range(0, total_track)]
							note_off_list = [[] for i in range(0, total_track)]

					end_delta_time = 400 * 50
					for i in range(2, len(track_string_list)):
						for j in track_string_list[i]:
							new_csv_string.append(j)
						new_csv_string.append(self.end_track_string(i, end_delta_time))
					new_csv_string.append("0, 0, End_of_file\n") # end of file string
					# print('NEW STRING')

					# data = pd.DataFrame(new_csv_string)
					# data.to_csv(csv_output_dir,index = False)

					midi_object = py_midicsv.csv_to_midi(new_csv_string)

					with open(
						self.output_file_dir + "New_" + self.atype + "_" + only_file_name, "wb"
					) as output_file:
						midi_writer = py_midicsv.FileWriter(output_file)
						midi_writer.write(midi_object)
						print("Good Midi File")

						success_num += 1


					# For Cheking Error Data, Represent to csv files
					if self.csv_printable:
						self.checking_csv(only_file_name)


	def checking_csv(self, only_file_name):
		csv_string = py_midicsv.midi_to_csv(self.output_file_dir + "New_" + self.atype + "_" + only_file_name)
		tmp_list = []
		for i in range(0, len(csv_string)):
			temp = np.array(csv_string[i].replace("\n", "").replace(" ", "").split(","))
			tmp_list.append(temp)
		data = pd.DataFrame(tmp_list)
		data.to_csv(self.csv_output_dir + 'New_' + only_file_name[:-4] + '.csv', header=False, index=False)
		print(".csv saved!")