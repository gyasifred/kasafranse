cnt = 0
max_sentence = 0 # placeholder for max sentence length
lang_1 = [] # placeholder for total number of words in a document
unique_list = [] # placeholder for unique words in  document
path = ''
lines = open(path) 
for line in lines:
    ln = line.strip().split()
    for word in ln:
        lang_1.append(word)
    if len(ln) > max_sentence:
         max_sentence = len(ln)
    else:
        max_sentence  = max_sentence 
    cnt +=1

for x in lang_1:
    if x not in unique_list:
        unique_list.append(x)
print(f'Total number of sentences: {cnt}')
print(f'Maximum sentence lenght: {max_sentence}')
print(f'Total number of words: {len(lang_1)}')
print(f'Total number of unique words: {len(unique_list)}')
    