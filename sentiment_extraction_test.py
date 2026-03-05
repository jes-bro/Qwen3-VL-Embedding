from pyabsa import AspectTermExtraction as ATEPC

# Initialize the aspect extractor
aspect_extractor = ATEPC.AspectExtractor('multilingual', auto_device=True)

# Perform aspect extraction on a sample sentence
result = aspect_extractor.predict(
    ['The person is doing chest compressions too slowly. They need to speed up. They have a good hand position though.']
)

result_opposite = aspect_extractor.predict(
    ['The person is doing great chest compressions. They are doing great. They have a rough hand position though.']
)

result_all_good = aspect_extractor.predict(
    ['The person is doing great chest compressions. They are doing great. They have a good hand position too.']
)

print(result[0])
# print(result[0]["tokens"][20])
# suppress output
bad_list = []
good_list = [] # should these be np arr? later

# Collect good and bad
for idx, token in enumerate(result[0]["aspect"]):
    if result[0]["sentiment"][idx] == 'Negative':
        bad_list.append(token)
    elif result[0]["sentiment"][idx] == 'Positive':
        good_list.append(token)

print(f'good list: {good_list}')
print(f'bad list: {bad_list}')

# embed the good and bad lists and store as separate vectors
