import streamlit as st
import openai
import pandas as pd
import os
import re
import datetime
from difflib import get_close_matches

# ---------------------------
# 설정 및 CSV 로딩
# ---------------------------
OCR_MODEL = "gpt-4o"         # 이미지 → 텍스트(OCR) 모델
TRANSFORM_MODEL = "o3-mini"   # 문제 변형 및 검증 모델

# OpenAI API 키 (실제 키로 교체)
openai.api_key = "YOUR_OPENAI_API_KEY_HERE"

@st.cache_data
def load_csv_files():
    df1 = pd.read_csv("data1.csv")  # data1에는 "tag", "concepts", "contents", "subject", "grade", "mj", "im", "index" 컬럼이 있음.
    df2 = pd.read_csv("data2.csv")
    try:
        df3 = pd.read_csv("data3.csv")
    except FileNotFoundError:
        st.warning("data3.csv 파일이 존재하지 않습니다. data3 기능은 비활성화됩니다.")
        df3 = None
    return df1, df2, df3

df_data1, df_data2, df_data3 = load_csv_files()

# ---------------------------
# 함수 정의
# ---------------------------
def get_allowed_and_disallowed_concepts_by_range(index_min: int, index_max: int):
    df_allowed = df_data1[(df_data1["index"] >= index_min) & (df_data1["index"] <= index_max)]
    allowed_concepts = set()
    for contents_str in df_allowed["contents"]:
        if isinstance(contents_str, str):
            allowed_concepts.update([c.strip() for c in contents_str.split(",")])
    df_disallowed = df_data1[(df_data1["index"] < index_min) | (df_data1["index"] > index_max)]
    disallowed_concepts = set()
    for contents_str in df_disallowed["contents"]:
        if isinstance(contents_str, str):
            disallowed_concepts.update([c.strip() for c in contents_str.split(",")])
    return list(allowed_concepts), list(disallowed_concepts)

def match_dimension(row_val, user_val):
    if row_val == "none":
        return True
    return (row_val == user_val)

def filter_guide_data(subject, grade, mj, im):
    filtered_guides = []
    for idx, row in df_data2.iterrows():
        row_subject = str(row.get("subject", "none"))
        row_grade   = str(row.get("grade", "none"))
        row_mj      = str(row.get("mj", "none"))
        row_im      = str(row.get("im", "none"))
        if match_dimension(row_subject, subject) and match_dimension(row_grade, grade) and \
           match_dimension(row_mj, mj) and match_dimension(row_im, im):
            filtered_guides.append(row["guide"])
    return filtered_guides

def ocr_with_gpt4o_mini(image_url):
    if not image_url:
        return ""
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Please extract all visible text from this image. "
                    "Then output the text as normal text (Korean) for non-mathematical parts, "
                    "but convert only the mathematical expressions into LaTeX by enclosing them in $...$. "
                    "Do not wrap the entire text in any LaTeX environment such as \\[...\\]. "
                    "For example, if the text says '-2 <= x <= 0', convert that to '$-2 \\leq x \\leq 0$', "
                    "and if the text says 'y = (x^2 + 6x)^2 + 4(x^2 + 6x) - 6', convert that to '$y = (x^2 + 6x)^2 + 4(x^2 + 6x) - 6$'. "
                    "Output only the final text."
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
    }
    try:
        response = openai.chat.completions.create(
            model=OCR_MODEL,
            messages=[user_message],
            temperature=1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"[OCR 오류] {e}")
        return ""

def transform_problem_text(original_problem_text, original_solution_text, n_problems,
                           index_min, index_max, target_tag, user_difficulty, user_instructions="", selected_concepts=""):
    if target_tag != "none" and target_tag not in df_data1["tag"].values:
        st.error(f"잘못된 tag: {target_tag} (data1에 존재하지 않음)")
        return ""
    if target_tag != "none":
        allowed_concepts, disallowed_concepts = get_allowed_and_disallowed_concepts_by_range(index_min, index_max)
    else:
        allowed_concepts, disallowed_concepts = [], []
    if target_tag != "none":
        try:
            row = df_data1.loc[df_data1["tag"] == target_tag].iloc[0]
            subject_val = str(row.get("subject", "none"))
            grade_val   = str(row.get("grade", "none"))
            mj_val      = str(row.get("mj", "none"))
            im_val      = str(row.get("im", "none"))
        except Exception as e:
            st.error(f"data1에서 tag {target_tag}를 찾지 못했습니다: {e}")
            return ""
    else:
        subject_val = grade_val = mj_val = im_val = ""
    if target_tag != "none":
        guide_list = filter_guide_data(subject_val, grade_val, mj_val, im_val)
        guide_text = "\n".join(guide_list)
    else:
        guide_text = ""
    data3_text = ""
    data3_difficulty_info = ""
    if df_data3 is not None and target_tag != "none":
        data3_rows = df_data3[df_data3["tag"] == target_tag]
        if not data3_rows.empty:
            # 난이도 분석: 숫자형으로 변환 후 평균, 최소, 최대 계산
            difficulties = pd.to_numeric(data3_rows["difficulty"], errors='coerce').dropna()
            if not difficulties.empty:
                avg_diff = difficulties.mean()
                min_diff = difficulties.min()
                max_diff = difficulties.max()
                data3_difficulty_info = f"데이터3 난이도 분석: 평균 {avg_diff:.2f}, 범위 {min_diff} ~ {max_diff}."
            for idx, drow in data3_rows.iterrows():
                sample_q   = str(drow.get("question", ""))
                difficulty = str(drow.get("difficulty", ""))
                qtype      = str(drow.get("type", ""))
                exp        = str(drow.get("exp", ""))
                data3_text += (
                    f"[예제]\n{sample_q}\n"
                    f"난이도: {difficulty}\n"
                    f"유형: {qtype}\n"
                    f"해설(참고): {exp}\n\n"
                )
        else:
            data3_text = "(해당 tag로 data3에서 찾을 수 있는 예제/추가정보가 없습니다.)"
    if original_solution_text:
        solution_part = f"\n\n원본 해설 (OCR 추출):\n{original_solution_text}\n\n" \
                        "위 해설의 형식과 스타일을 참고하여, 변형된 문제의 해설도 유사한 양식으로 작성해 주세요.\n"
    else:
        solution_part = "\n\n(해설 이미지 없음: 원본 해설 없이 진행)\n"
    prompt = f"""
원본 문제 (OCR 추출):
{original_problem_text}
{solution_part}

인덱스 범위: {index_min} ~ {index_max}
(이 범위 안에서 배운 개념만 사용 가능합니다)

사용자 요구 난이도: {user_difficulty}
{data3_difficulty_info}

선택된 태그의 핵심 개념 (원본 문제에서 사용됨): {selected_concepts}
※ 이 내용은 해당 tag의 "contents" 컬럼의 실제 핵심개념입니다.
변형 문제에서도 반드시 이 핵심 개념을 반영해야 합니다.

추가 지시사항:
{user_instructions}

※ 아래 문구를 참고하여, 변형된 문제와 해설에서도 수식 부분은 LaTeX 형식($...$)으로 변환해 주세요.
"Please extract all visible text from this image. Then output the text as normal text (Korean) for non-mathematical parts, but convert only the mathematical expressions into LaTeX by enclosing them in $...$. Do not wrap the entire text in any LaTeX environment such as \\[...\\]. For example, if the text says '-2 <= x <= 0', convert that to '$-2 \\leq x \\leq 0$', and if the text says 'y = (x^2 + 6x)^2 + 4(x^2 + 6x) - 6', convert that to '$y = (x^2 + 6x)^2 + 4(x^2 + 6x) - 6$'."

허용 개념:
{", ".join(allowed_concepts)}

비허용 개념:
{", ".join(disallowed_concepts)}

(참고) data2 가이드:
{guide_text}

(참고) data3의 예시/해설:
{data3_text}

중요: data3 정보는 참고용일 뿐입니다. 원본 문제를 우선하여 변형을 진행해주세요.

지시사항:
1) 원본 문제의 구조를 크게 훼손하지 않고, 숫자(계수)나 조건만 적절히 변경하되
   - 반드시 핵심 개념을 포함하고
   - 수학적 추론 과정을 담은 풀이와 정답(해설)을 포함
   - 난이도 {user_difficulty} 정도로 구성
2) 각 최종 문제는 "문제:"로 시작, 해설은 "해설:"로 시작.
3) 마지막에 [추출된 문제 목록] 라벨 아래, 생성된 문제들을 줄바꿈으로 구분하여 나열.
4) 총 {n_problems} 문제 작성.
""".strip()
    try:
        response = openai.chat.completions.create(
            model=TRANSFORM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"[o3-mini] 변형 API 오류: {e}")
        return ""

def extract_problems_and_explanations(response_text):
    if "[추출된 문제 목록]" in response_text:
        content = response_text.split("[추출된 문제 목록]", 1)[1].strip()
    else:
        content = response_text.strip()
    blocks = re.split(r"\n\s*\n", content)
    result = []
    for block in blocks:
        if "문제:" in block and "해설:" in block:
            parts = block.split("해설:", 1)
            question_part = parts[0].strip()
            explanation_part = parts[1].strip()
            if question_part.startswith("문제:"):
                question_part = question_part[len("문제:"):].strip()
            result.append({"question": question_part, "exp": explanation_part})
    return result

def verify_problem(problem, exp):
    prompt = f"""
다음 문제와 해설이 수학적, 논리적으로 이상이 없는지 검증해 주세요.
문제: {problem}
해설: {exp}

검증 결과가 모두 이상이 없으면 "Verified"라고 출력하고, 그렇지 않으면 "Not Verified: [오류 내용]"으로 답해 주세요.
"""
    try:
        response = openai.chat.completions.create(
            model=TRANSFORM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"[검증 오류] {e}")
        return "Not Verified: API Error"

def save_problems_to_csv(tag_val, subject_val, grade_val, mj_val, im_val, problems, user_difficulty, output_csv):
    if not problems:
        st.info("문제가 없으므로 CSV에 저장하지 않습니다.")
        return
    rows = []
    for item in problems:
        rows.append({
            "tag": tag_val,
            "subject": subject_val,
            "grade": grade_val,
            "mj": mj_val,
            "im": im_val,
            "difficulty": user_difficulty,
            "newquestion": item["question"],
            "exp": item["exp"]
        })
    df_out = pd.DataFrame(rows)
    if not os.path.exists(output_csv):
        df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    else:
        df_out.to_csv(output_csv, mode='a', header=False, index=False, encoding="utf-8-sig")
    st.success(f"[CSV 저장] '{tag_val}' 변형 문제를 '{output_csv}' 파일에 저장했습니다.")

# ---------------------------
# Streamlit UI 구성
# ---------------------------
st.title("문제 변형 및 해설 생성")

st.sidebar.header("입력 항목")
problem_image_url = st.sidebar.text_input("원본 문제 이미지 URL", "")
solution_image_url = st.sidebar.text_input("원본 해설 이미지 URL (없으면 비워두세요)", "")
n_problems = st.sidebar.number_input("생성할 문제 수", min_value=1, step=1, value=3)
user_instructions = st.sidebar.text_area("문제 변형 시 추가 지시사항 (없으면 비워두세요)", "")
user_difficulty = st.sidebar.select_slider("문제 난이도 (0~5)", options=list(range(6)), value=3)

# 태그 검색 및 추천 (표시에는 concepts, 실제 프롬프트에는 contents 사용)
st.sidebar.subheader("태그 검색")
user_tag_search = st.sidebar.text_input("검색어 입력", "")
# tag 옵션은 "tag / concepts" 형태로 표시 (여기서 concepts는 UI 표시용)
tag_options = df_data1.apply(lambda row: f"{row['tag']} / {row['concepts']}", axis=1).tolist()
if user_tag_search:
    filtered_df = df_data1[df_data1["concepts"].str.contains(user_tag_search, case=False, na=False)]
    if not filtered_df.empty:
        recommended_options = filtered_df.apply(lambda row: f"{row['tag']} / {row['concepts']}", axis=1).tolist()
        st.sidebar.write("추천 태그:")
        selected_option = st.sidebar.selectbox("추천 태그 선택", recommended_options)
    else:
        st.sidebar.write("추천 태그 없음. 전체 목록에서 선택하세요.")
        selected_option = st.sidebar.selectbox("태그와 개념 선택", tag_options)
else:
    selected_option = st.sidebar.selectbox("태그와 개념 선택", tag_options)

selected_tag = selected_option.split(" / ")[0]
# UI 표시용 concepts (사용자에게 보여줄 내용)
display_concepts = selected_option.split(" / ")[1]
# 실제 프롬프트에 사용할 핵심개념은 해당 tag의 "contents" 컬럼에서 가져옴.
actual_contents = df_data1.loc[df_data1["tag"] == selected_tag, "contents"].iloc[0]

# 인덱스 범위: 최소값은 1, 최대값은 선택 옵션
st.sidebar.subheader("인덱스 범위 선택")
index_option = st.sidebar.radio("최대 인덱스 선택 옵션", ("선택된 태그의 최대 인덱스", "전체 데이터의 최대 인덱스"))
if index_option == "선택된 태그의 최대 인덱스":
    if selected_tag != "none":
        available_indices = sorted(df_data1[df_data1["tag"] == selected_tag]["index"].unique())
    else:
        available_indices = sorted(df_data1["index"].unique())
else:
    available_indices = sorted(df_data1["index"].unique())
index_max = st.sidebar.selectbox("최대 인덱스 선택", available_indices, index=len(available_indices)-1)
index_min = 1

st.write("### 입력 정보")
st.write("문제 이미지 URL:", problem_image_url)
st.write("해설 이미지 URL:", solution_image_url if solution_image_url else "없음")
st.write("생성할 문제 수:", n_problems)
st.write("문제 난이도:", user_difficulty)
st.write("추가 지시사항:", user_instructions)
st.write("선택된 인덱스 범위:", index_min, "~", index_max)
st.write("선택된 태그:", selected_tag)
st.write("UI에 표시되는 개념 (concepts):", display_concepts)
st.write("실제 프롬프트에 사용할 핵심 개념 (contents):", actual_contents)

if st.button("문제 변형 생성"):
    with st.spinner("문제 이미지에서 텍스트 추출 중..."):
        original_problem_text = ocr_with_gpt4o_mini(problem_image_url)
    if not original_problem_text:
        st.error("문제 이미지에서 텍스트를 추출하지 못했습니다.")
        st.stop()
    original_solution_text = ""
    if solution_image_url:
        with st.spinner("해설 이미지에서 텍스트 추출 중..."):
            original_solution_text = ocr_with_gpt4o_mini(solution_image_url)
    with st.spinner("문제 변형 중..."):
        transform_response = transform_problem_text(
            original_problem_text=original_problem_text,
            original_solution_text=original_solution_text,
            n_problems=n_problems,
            index_min=index_min,
            index_max=index_max,
            target_tag=selected_tag,
            user_difficulty=user_difficulty,
            user_instructions=user_instructions,
            selected_concepts=actual_contents
        )
    st.write("### 변형 모델 응답")
    st.code(transform_response)
    problems = extract_problems_and_explanations(transform_response)
    
    # 검증 단계: 각 문제와 해설을 GPT 모델로 검증하여 "Verified"인 항목만 필터링
    verified_problems = []
    if problems:
        st.write("### 검증 진행 중...")
        for item in problems:
            verification_result = verify_problem(item["question"], item["exp"])
            st.write(f"검증 결과: {verification_result} - 문제: {item['question'][:50]}...")
            if verification_result.strip().lower() == "verified":
                verified_problems.append(item)
    else:
        st.warning("추출된 문제 목록을 찾지 못했거나 문제가 없습니다.")
    
    if verified_problems:
        st.write("### 검증 통과한 문제 목록")
        for item in verified_problems:
            st.write("**문제:**", item["question"])
            st.write("**해설:**", item["exp"])
    else:
        st.warning("검증을 통과한 문제 없음.")
    
    # CSV 저장: 파일명은 "실행일자_[선택된 tag]_[tag에 해당하는 contents].csv"
    if selected_tag != "none" and verified_problems:
        try:
            row = df_data1.loc[df_data1["tag"] == selected_tag].iloc[0]
            subject_val = str(row.get("subject", "none"))
            grade_val = str(row.get("grade", "none"))
            mj_val = str(row.get("mj", "none"))
            im_val = str(row.get("im", "none"))
        except Exception as e:
            st.error(f"[CSV 저장 오류] data1에서 tag 검색 실패: {e}")
            subject_val = grade_val = mj_val = im_val = "none"
        output_csv = f"{datetime.datetime.now().strftime('%Y%m%d')}_{selected_tag}_{actual_contents}.csv"
        save_problems_to_csv(
            tag_val=selected_tag,
            subject_val=subject_val,
            grade_val=grade_val,
            mj_val=mj_val,
            im_val=im_val,
            problems=verified_problems,
            user_difficulty=user_difficulty,
            output_csv=output_csv
        )
