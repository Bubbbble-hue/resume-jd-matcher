import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re

# 1. 加载环境变量 (读取 .env 里的 DeepSeek API Key)
load_dotenv()

# 2. 初始化 DeepSeek 客户端 (兼容 OpenAI 格式)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 3. 核心分析函数 (完全复用你之前的逻辑)
def analyze_resume(resume_text, jd_text):
    if not resume_text or not jd_text:
        return "请同时填写简历和职位描述。", "", "", ""
    
    prompt = f"""
    你是一位资深的技术招聘专家。请基于以下“简历”和“职位描述(JD)”，对候选人进行匹配度分析。
    
    ## 职位描述 (JD):
    {jd_text}
    
    ## 简历:
    {resume_text}
    
    ## 分析要求:
    请严格按照以下JSON格式输出分析结果，不要添加任何额外的解释文字，只输出纯JSON：
    {{
        "match_score": 匹配分数 (0-100的整数),
        "matched_skills": ["匹配的技能点1", "匹配的技能点2"],
        "missing_skills": ["缺失的技能点1", "缺失的技能点2"],
        "suggestions": ["修改建议1", "修改建议2"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.choices[0].message.content
        
        # 提取 JSON 结果
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            score = f"📊 匹配度评分: {result['match_score']}/100"
            matched = "\n".join([f"- {s}" for s in result['matched_skills']])
            missing = "\n".join([f"- {s}" for s in result['missing_skills']])
            suggestions = "\n".join([f"- {s}" for s in result['suggestions']])
            return score, matched, missing, suggestions
        else:
            return "解析失败", result_text, "", ""
    except Exception as e:
        return f"API 报错: {str(e)}", "", "", ""

# 4. 使用 Gradio 搭建界面
with gr.Blocks(title="简历-JD 匹配分析助手") as demo:
    gr.Markdown("# 📄 简历-JD 匹配分析助手")
    gr.Markdown("粘贴你的简历和职位描述，AI 将为你分析匹配度并给出修改建议。")
    
    with gr.Row():
        with gr.Column():
            resume_input = gr.Textbox(label="📌 你的简历", lines=15)
        with gr.Column():
            jd_input = gr.Textbox(label="🎯 职位描述 (JD)", lines=15)
            
    btn = gr.Button("开始分析", variant="primary")
    
    gr.Markdown("## 分析结果")
    score_output = gr.Markdown(label="评分")
    
    with gr.Row():
        matched_output = gr.Markdown(label="✅ 匹配的技能")
        missing_output = gr.Markdown(label="❌ 缺失的技能")
        
    suggestions_output = gr.Markdown(label="💡 修改建议")
    
    # 绑定按钮点击事件
    btn.click(
        fn=analyze_resume,
        inputs=[resume_input, jd_input],
        outputs=[score_output, matched_output, missing_output, suggestions_output]
    )

# 5. 启动应用
if __name__ == "__main__":
    demo.launch()