# PLEASE run in 'split' folder


import os
import random

INPUT_PATH = '/data/inputs/'
composer = []
composer_count = [] # Total segment num of each composer
composer_midi_count = [] # Total midi folder num of each composer
train_idx = []

train_percentage = 0.9

random.seed(333) # change this

for folder in os.listdir(INPUT_PATH):
	if folder == 'name_id_map.csv': continue
	composer.append(folder.replace('composer', ''))

	composer_fold = INPUT_PATH + folder + '/'
	count = 0
	midi_count = 0
	for midi_f in os.listdir(composer_fold):
		# print(midi_f)
		midi_count += 1

		midi_fold = composer_fold + midi_f
		for file in os.listdir(midi_fold):
			count += 1

	# print(count)
	composer_count.append(count)
	composer_midi_count.append(midi_count)

print("## composer index:")
print(composer)
print("## composer's total counts:")
print(composer_count)
print("## composer's total midi counts:")
print(composer_midi_count)

print("total:", sum(composer_count))
print("train num:", int(sum(composer_count) * train_percentage))
print("test num:", int(sum(composer_count) * (1 - train_percentage)))
print()


for count in composer_count:
	# print(list(range(0, count)))

	idxlist = random.sample(list(range(0, count)), int(count * train_percentage))
	train_idx.append(idxlist)


train_file = open('/data/split/train.txt', 'w')
test_file = open('/data/split/test.txt', 'w')
composer_idx = 0 # 0 ~ 14
total_train_count = 0
total_test_count = 0
for folder in os.listdir(INPUT_PATH):
	if folder == 'name_id_map.csv': continue
	composer_fold = INPUT_PATH + folder + '/'
	idx = -1
	for midi_f in os.listdir(composer_fold):
		# print(midi_f)
		midi_fold = composer_fold + midi_f + '/'
		for file in os.listdir(midi_fold):
			idx += 1

			if idx in train_idx[composer_idx]:
				# print(midi_fold + file)
				train_file.write(midi_fold + file + '\n')
				total_train_count += 1
				
			else:
				test_file.write(midi_fold + file + '\n')
				total_test_count += 1
				

	composer_idx += 1

print('## After split:')
print('total train count:', total_train_count)
print('total test count:', total_test_count)


train_file.close()
test_file.close()

