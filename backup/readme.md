benchmark and test script

main_v02.py函数说明：

No.	Function 	Note

1	import LLM_mtl_v02

LLM_mtl_v02.init()	模型加载初始化。启动UI后只需要运行一次，模型会放在内存里

2	LLM_mtl_v02.predict	输入：

•	audio_input: 用户输入语音mp4/wav

•	llm_model_select="chatglm3-6b": 用户选择的大模型

•	max_length=512: 最大输出长度

•	save_audio=Ture: 输出的文字保存成音频在audiollm/audio_cache

输出：

•	text_in: 用户输入的语音转成文字

•	text_out: 大模型输出的文字

•	llm_ms_first_token_list: First Latency

•	llm_ms_after_token_list: After Latency

•	sr_latency: 语音识别耗时

•	wav_path: 输出的文字转成音频保存在audiollm/audio_cache

3	LLM_mtl_v02.predict_user	输入：

•	user_input="你是谁": 用户输入文字

•	llm_model_select="chatglm3-6b": 用户选择的大模型

•	max_length=512: 最大输出长度

•	save_audio=Ture: 输出的文字保存成音频在audiollm/audio_cache

输出：

•	text_in: 用户输入的文字

•	text_out: 大模型输出的文字

•	llm_ms_first_token_list: First Latency

•	llm_ms_after_token_list: After Latency

•	wav_path: 输出的文字转成音频保存在audiollm/audio_cache

4	LLM_mtl_v02.stop_play()	•	停止大模型输出文字，

•	停止保存音频，

•	删除audiollm/audio_cache里的所有音频
