import json
from pyabsa import AspectTermExtraction as ATEPC

print("got here")
# Initialize the aspect extractor
config = ATEPC.ATEPCConfigManager.get_atepc_config_english()

# Adjust confidence threshold (e.g., 0.7)
config.confidence_threshold = 0.1
aspect_extractor = ATEPC.AspectExtractor('english', auto_device=True, config=config)
print("in file")
# Perform aspect extraction on a sample sentence
result_nov = aspect_extractor.predict(
    ["Both the rate and the depth of her compressions are not appropriate. The rate needs to be increased to 100-120 beats per minute and compression depth should be at least 2 inches or 5 cm."]
)

print(f'nov result: {result_nov}')

result_all_good = aspect_extractor.predict(
    [" His hand placement is excellent, right below the nipple line in the middle of the sternum. This is very good. His body posture is also excellent with his elbows straight and he's hinging at the hips to deliver his compressions.",]
)

print(f'all good result: {result_all_good}')

result_opposite = aspect_extractor.predict(
    [" Again, rate of compressions is good. I think his hand placement needs to be slightly higher on the sternum and the depth looks adequate. The compression rate is between 100 and 120 beats per minute. The compression depth is at least 2 inches."]
)

print(f'hand placement result: {result_opposite}')

# " Her compression rate is good and it does appear that she's allowing for full recoil of the chest in between each compression. That's essential because the heart needs to fill with blood, so you don't want to be continually compressing the chest. You need to allow those full breaks in between each compression for the heart to fill up with blood. Her compression depth could be a little bit better. It should be about two inches and again positioning will make that movement and that energy more efficient. Her hand placement is good and her compression rate is pretty good. She does appear to be delivering the compressions at at least a hundred beats per minute, which is the ideal range of 100 to 120 for effective CPR.",
result_need_better_depth = aspect_extractor.predict(
    [" Her compression rate is good and it does appear that she's allowing for full recoil of the chest in between each compression. That's essential because the heart needs to fill with blood, so you don't want to be continually compressing the chest. You need to allow those full breaks in between each compression for the heart to fill up with blood. Her compression depth could be a little bit better. It should be about two inches and again positioning will make that movement and that energy more efficient. Her hand placement is good and her compression rate is pretty good. She does appear to be delivering the compressions at at least a hundred beats per minute, which is the ideal range of 100 to 120 for effective CPR."]
)

print(f'needs better deoth result: {result_need_better_depth}')

# print(result[0]["tokens"][20])
# suppress output
 # should these be np arr? later

# Collect good and bad
def get_good_and_bad_list(result):
    bad_list = []
    good_list = []
    for idx, token in enumerate(result[0]["aspect"]):
        # print(f'result at idx: {result[0]["aspect"][idx]}')
        if result[0]["sentiment"][idx] == 'Negative':
            bad_list.append(token)
        elif result[0]["sentiment"][idx] == 'Positive':
            good_list.append(token)
    
    print(f'good list: {good_list}')
    print(f'bad list: {bad_list}')
    return ", ".join(sorted(good_list)), ", ".join(sorted(bad_list))

print("nove good and bad:")
good_list_nov, bad_list_nov = get_good_and_bad_list(result_nov)
print("slightly higher hands needed good and bad:")
good_list_intexp, bad_list_intexp = get_good_and_bad_list(result_opposite)
print("expert good and bad:")
good_list_exp, bad_list_exp = get_good_and_bad_list(result_all_good)
print("better depth needed good and bad:")
good_list_almost, bad_list_almost = get_good_and_bad_list(result_need_better_depth)

good_and_bads_per_vid = {'novice': {'good': good_list_nov, 'bad': bad_list_nov}, 'notbad': {'good': good_list_intexp, 'bad:': bad_list_intexp}, 'almost': {'good': good_list_exp, 'bad': bad_list_exp}, 'betterdepth': {'good': good_list_almost, 'bad':bad_list_almost}}

output_path = '/home/jess/Qwen3-VL-Embedding/goodbadoutputs.json'
with open(output_path, 'w') as file:
    json.dump(good_and_bads_per_vid, file)

# fine cause this will be pre-processing step on dataset and will only need to be called once for new novice clip maybe more but for now 1

