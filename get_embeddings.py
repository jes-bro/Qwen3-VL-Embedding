import json
from pyabsa import AspectTermExtraction as ATEPC
from ollama import chat
from ollama import ChatResponse
# print("got here")
# Initialize the aspect extractor
config = ATEPC.ATEPCConfigManager.get_atepc_config_english()

# Adjust confidence threshold (e.g., 0.7)
# config.confidence_threshold = 0.1
aspect_extractor = ATEPC.AspectExtractor('english', auto_device=True, config=config)
print("in file")
# Perform aspect extraction on a sample sentence
# result_nov = aspect_extractor.predict(
#     ["Both the rate and the depth of her compressions are not appropriate. The rate needs to be increased to 100-120 beats per minute and compression depth should be at least 2 inches or 5 cm."]
# )

# print(f'nov result: {result_nov}')

# result_all_good = aspect_extractor.predict(
#     [" His hand placement is excellent, right below the nipple line in the middle of the sternum. This is very good. His body posture is also excellent with his elbows straight and he's hinging at the hips to deliver his compressions.",]
# )

# print(f'all good result: {result_all_good}')

# result_opposite = aspect_extractor.predict(
#     [" Again, rate of compressions is good. I think his hand placement needs to be slightly higher on the sternum and the depth looks adequate. The compression rate is between 100 and 120 beats per minute. The compression depth is at least 2 inches."]
# )

# print(f'hand placement result: {result_opposite}')

# " Her compression rate is good and it does appear that she's allowing for full recoil of the chest in between each compression. That's essential because the heart needs to fill with blood, so you don't want to be continually compressing the chest. You need to allow those full breaks in between each compression for the heart to fill up with blood. Her compression depth could be a little bit better. It should be about two inches and again positioning will make that movement and that energy more efficient. Her hand placement is good and her compression rate is pretty good. She does appear to be delivering the compressions at at least a hundred beats per minute, which is the ideal range of 100 to 120 for effective CPR.",
# result_need_better_depth = aspect_extractor.predict(
#     [" Her compression rate is good and it does appear that she's allowing for full recoil of the chest in between each compression. That's essential because the heart needs to fill with blood, so you don't want to be continually compressing the chest. You need to allow those full breaks in between each compression for the heart to fill up with blood. Her compression depth could be a little bit better. It should be about two inches and again positioning will make that movement and that energy more efficient. Her hand placement is good and her compression rate is pretty good. She does appear to be delivering the compressions at at least a hundred beats per minute, which is the ideal range of 100 to 120 for effective CPR."]
# )

# print(f'needs better deoth result: {result_need_better_depth}')

# print(result[0]["tokens"][20])
# suppress output
 # should these be np arr? later

# Collect good and bad
def get_good_and_bad_list(annotation):
    result = aspect_extractor.predict(
    [annotation]
    )
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

input_json_path = "/home/jess/sm/temp_cpr_sub_result.json"

with open(input_json_path, "r") as f:
    subtasks = json.load(f)

for subtask in subtasks:
    print(subtask)
    for video_name in subtasks[subtask]:
        timestamp_dicts = subtasks[subtask][video_name]["time_stamps_file_paths_poses"]
        print(timestamp_dicts.keys())
        for timestamp in timestamp_dicts:
            if "commentary" in subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp].keys():
                commentary_at_timestamp = subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["commentary"]
                commentary_text = "".join(commentary_at_timestamp)
                response_summary: ChatResponse = chat(model='gemma3', messages=[
                {
                    'role': 'user',
                    'content': f'Summarize the skills that the person did well and poorly at based on the expert commentary: {commentary_text}. Plain text only. No Markdown or formatting or new lines. Be objective. Include negative stuff too. If there\'s none of good or bad, just leave it out. If the text is neutral, leave that out as well.',
                    'format' : 'json'
                },
                ])
                response_summary_text = response_summary.message.content
                response_negative: ChatResponse = chat(model='gemma3', messages=[
                {
                    'role': 'user',
                    'content': f'You are the best sentiment analysis program in the world. Output a comma separated list of 1-3 words the skills that the person did poorly at based on the expert commentary: {response_summary_text}. Plain text only. No Markdown or formatting or new lines. Adjectives should be paired with verbs if used. Ideas should be contained in one string. If there\'s none, that\'s ok. Just a comma separated list.',
                    'format' : 'json'
                },
                ])
                response_positive: ChatResponse = chat(model='gemma3', messages=[
                {
                    'role': 'user',
                    'content':f'You are the best sentiment analysis program in the world. Output a comma separated list of 1-3 words the skills that the person did well at based on the expert commentary: {response_summary_text}. Plain text only. No Markdown or formatting or new lines. Adjectives should be paired with verbs if used. Ideas should be contained in one string. If there\'s none, that\'s ok. Just a comma separated list.',
                    'format' : 'json'
                },
                ])
                good = response_positive.message.content.lower()
                bad = response_negative.message.content.lower()
                subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["commentary"] = response_summary_text
                subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["good"] = good.split(", ")
                subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["bad"] = bad.split(", ")
                print(f"original text: {commentary_text}")
                print(f"summary: {response_summary_text}")
                print(f"lists: good: {good}")
                print(f"lists: bad: {bad}")
                # good_list, bad_list = get_good_and_bad_list(summary)
                # subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["good_list"] = good_list
                # subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["bad_list"] = bad_list
                # print(f"original text: {commentary_text}")

dest_path = "/home/jess/Qwen3-VL-Embedding/goodsbadsfromsumm.json" # "/home/jess/Qwen3-VL-Embedding/goodsbads.json"
with open(dest_path, 'w') as f:
    json.dump(subtasks, f)

# print("nove good and bad:")
# good_list_nov, bad_list_nov = get_good_and_bad_list(result_nov)
# print("slightly higher hands needed good and bad:")
# good_list_intexp, bad_list_intexp = get_good_and_bad_list(result_opposite)
# print("expert good and bad:")
# good_list_exp, bad_list_exp = get_good_and_bad_list(result_all_good)
# print("better depth needed good and bad:")
# good_list_almost, bad_list_almost = get_good_and_bad_list(result_need_better_depth)

# good_and_bads_per_vid = {'novice': {'good': good_list_nov, 'bad': bad_list_nov}, 'notbad': {'good': good_list_intexp, 'bad:': bad_list_intexp}, 'almost': {'good': good_list_exp, 'bad': bad_list_exp}, 'betterdepth': {'good': good_list_almost, 'bad':bad_list_almost}}

# output_path = '/home/jess/Qwen3-VL-Embedding/goodbadoutputs.json'
# with open(output_path, 'w') as file:
#     json.dump(good_and_bads_per_vid, file)

# fine cause this will be pre-processing step on dataset and will only need to be called once for new novice clip maybe more but for now 1

