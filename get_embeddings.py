import json
from pyabsa import AspectTermExtraction as ATEPC

print("got here")
# Initialize the aspect extractor
aspect_extractor = ATEPC.AspectExtractor('multilingual', auto_device=True)
print("in file")
# Perform aspect extraction on a sample sentence
result_nov = aspect_extractor.predict(
    ['The person is doing chest compressions too slowly. They need to speed up. They have a good hand position though.']
)

result_opposite = aspect_extractor.predict(
    ['The person is doing great chest compressions. They are doing great. They have a rough hand position though.']
)

result_all_good = aspect_extractor.predict(
    ['The person is doing great chest compressions. They are doing great. They have a good hand position too.']
)

# print(result[0]["tokens"][20])
# suppress output
bad_list = []
good_list = [] # should these be np arr? later

# Collect good and bad
def get_good_and_bad_list(result):
    for idx, token in enumerate(result[0]["aspect"]):
        print(f'result at idx: {result[0]["aspect"][idx]}')
        if result[0]["sentiment"][idx] == 'Negative':
            bad_list.append(token)
        elif result[0]["sentiment"][idx] == 'Positive':
            good_list.append(token)
    
    print(f'good list: {good_list}')
    print(f'bad list: {bad_list}')
    return ", ".join(sorted(good_list)), ", ".join(sorted(bad_list))


good_list_nov, bad_list_nov = get_good_and_bad_list(result_nov)
good_list_intexp, bad_list_intexp = get_good_and_bad_list(result_opposite)
good_list_exp, bad_list_exp = get_good_and_bad_list(result_all_good)

good_and_bads_per_vid = {'vid_1': {'good': good_list_nov, 'bad': bad_list_nov}, 'vid_2': {'good': good_list_intexp, 'bad:': bad_list_intexp}, 'vid_3': {'good': good_list_exp, 'bad': bad_list_exp}}

output_path = '/home/jess/Qwen3-VL-Embedding/goodbadoutputs.json'
with open(output_path, 'w') as file:
    json.dump(good_and_bads_per_vid, file)

# fine cause this will be pre-processing step on dataset and will only need to be called once for new novice clip maybe more but for now 1

