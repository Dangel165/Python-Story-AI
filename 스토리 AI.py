import streamlit as st
import torch
import pandas as pd
import io
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time


# --- 1. 모델 로드 및 캐시 처리 ---
@st.cache_resource
def load_model():
    """모델과 토크나이저를 로드하고 캐시하는 함수"""
    model_name = "skt/kogpt2-base-v2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        st.sidebar.success(f"연산 장치: {device}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        st.stop()


tokenizer, model, device = load_model()


# --- 2. 텍스트 생성 함수 
def generate_story(prompt, max_output_tokens, temperature, top_k, top_p, penalty, bad_words_list=None):
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    bad_words_ids = []
    if bad_words_list:
        for word in bad_words_list:
            if word.strip():
                ids = tokenizer.encode(word.strip(), add_special_tokens=False)
                if ids:
                    bad_words_ids.append(ids)

    output_sequences = model.generate(
        input_ids=input_ids,
        # 안정화를 위해 max_new_tokens 사용
        max_new_tokens=max_output_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=penalty,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        bad_words_ids=bad_words_ids if bad_words_ids else None
    )

    generated_text = tokenizer.decode(output_sequences[0].tolist(), skip_special_tokens=True)
    return generated_text


# --- 3. 세션 상태 초기화 및 데이터 로드 ---
if 'story_data' not in st.session_state:
    st.session_state.story_data = {
        'protagonist_name': "김서아",
        'protagonist_age': "24살의 주니어 백엔드 개발자",
        'story_genre': "판타지",
        'story_background': "서울의 복잡한 핀테크 회사 사무실",
        'start_prompt': "",
        'story_output': "",
        'narrative_style': "3인칭 (과거형)",
        'persona_keywords': "냉소적, 논리적",
        'bad_words': "컴퓨터, 바보, 멍청이",
        'required_plot': "주인공은 사실 엄청난 마력을 숨기고 있다.",
        'emotional_tone': 0,  # -5 to +5
        'sentence_structure': "보통 (서술적)",  # 짧고 간결 | 보통 | 길고 복잡
    }

if 'secondary_characters' not in st.session_state:
    st.session_state.secondary_characters = pd.DataFrame({
        '역할': ['조연', '악당'],
        '이름': ['이수호', '강태오'],
        '성격 키워드': ['친절함, 낙천적', '교활함, 야심적'],
        '주인공과의 관계': ['동료 개발자', '라이벌 회사 CEO']
    })

default_start = (
    f"{st.session_state.story_data['protagonist_name']}({st.session_state.story_data['protagonist_age']})는 "
    f"{st.session_state.story_data['story_background']}에서 근무한다. "
    "그녀는 오늘 아침 모니터에 알 수 없는 경고창이 뜬 것을 확인했다."
)
if not st.session_state.story_data['start_prompt']:
    st.session_state.story_data['start_prompt'] = default_start


def load_csv_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("CSV 파일이 비어 있습니다.")
            return
        row = df.iloc[0].to_dict()
        for key in st.session_state.story_data.keys():
            if key in row:
                st.session_state.story_data[key] = row[key]
        st.success("설정 불러오기 성공!")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")


# --- 4. 5단 구성 생성 함수 ---
def generate_five_stages_mode(base_prompt, max_output_tokens, temp, top_k, top_p, penalty, bad_words_list):
    """
    발단-전개-위기-절정-결말 5단계로 나누어 스토리를 순차적으로 생성합니다.
    """
    stages = ["발단", "전개", "위기", "절정", "결말"]
    full_story_parts = {}
    current_context = ""

    st.subheader("🗺️ 5단 구성 진행 중...")
    progress_bar = st.progress(0)

    for i, stage in enumerate(stages):
        # 5단 구성은 각 단계별 생성 길이를 150으로 고정하여 안정성을 확보합니다.
        stage_output_tokens = 150

        if i == 0:
            stage_instruction = f"이야기를 {base_prompt}로 시작하여 {stage} 단계를 70단어 내외로 작성해줘."
        else:
            stage_instruction = f"현재까지의 스토리: {current_context.strip()} 다음으로, {stage} 단계를 100단어 내외로 작성해줘."

        stage_output = generate_story(
            stage_instruction, stage_output_tokens, temp, top_k, top_p, penalty, bad_words_list
        )

        story_part = stage_output.replace(stage_instruction, "").strip()
        story_part = re.split(r'[.?!]\s*[A-Z가-힣]', story_part, 1)[0]

        full_story_parts[stage] = story_part
        current_context += " " + story_part

        st.info(f"✅ {stage} 완료")
        progress_bar.progress((i + 1) / len(stages))

    return "\n\n".join([f"### {stage}\n{text}" for stage, text in full_story_parts.items()])


# --- 5. 최종 프롬프트 구성 함수 (강화) ---
def build_final_prompt(user_prompt):
    """모든 제어 옵션과 캐릭터 정보를 통합하여 최종 프롬프트를 구성합니다."""

    char_info = []
    if not st.session_state.secondary_characters.empty:
        for index, row in st.session_state.secondary_characters.iterrows():
            if not pd.isna(row['이름']) and row['이름'].strip():
                char_info.append(f"[{row['역할']}-{row['이름']}: {row['성격 키워드']} / 관계: {row['주인공과의 관계']}]")

    current_tone_value = st.session_state.story_data['emotional_tone']
    tone_str = str(current_tone_value)
    if current_tone_value == 5:
        tone_str = "매우 밝고 희망적"
    elif current_tone_value == -5:
        tone_str = "매우 어둡고 비극적"

    control_info = (
            f"[장르: {st.session_state.story_data['story_genre']} / 시점: {st.session_state.story_data['narrative_style']}] "
            f"[주인공 성격: {st.session_state.story_data['persona_keywords']}] "
            f"[필수 플롯: {st.session_state.story_data['required_plot']}] "
            f"[감성 톤: {tone_str} / 문체: {st.session_state.story_data['sentence_structure']}] "
            + (" ".join(char_info) if char_info else "")
    )

    final_prompt = f"{control_info} 다음 설정을 기반으로 이야기를 시작해 줘. {user_prompt}"
    return final_prompt


# ================= GUI 화면 구성 =================

st.title("📚 스토리 생성 AI")

st.sidebar.header("📁 불러오기 / 설정")
uploaded_file = st.sidebar.file_uploader("CSV 설정 불러오기", type="csv")
if uploaded_file:
    load_csv_data(uploaded_file)

with st.sidebar.expander("❓ 기능 도움말"):
    st.markdown("""
    ---
    ### 🎬 생성 모드
    * **일반 생성:** 설정한 문구로 이야기를 쭉 이어가는 기본 모드입니다.
    * **5단 구성:** 발단-전개-위기-절정-결말 순서에 따라 체계적으로 스토리를 만듭니다.

    ---
    ### ⚙️ 고급 제어 기능
    * **멀티 캐릭터 설정:** 주인공 외의 **조연, 악당** 정보를 입력하면 AI가 그 역할을 인지합니다.
    * **감성 톤 슬라이더:** (-5=어둡고 비극적, 5=밝음)으로 스토리의 **전체적인 분위기**를 조절합니다.
    * **Top K/P:** 파라미터 값이 높을수록 **다양한 단어**를 선택하여 창의성이 높아집니다.
    """)

# --- 메인 입력창 (주인공) ---
st.header("👤 스토리 핵심 설정 (주인공)")
col1, col2 = st.columns(2)
with col1:
    st.session_state.story_data['protagonist_name'] = st.text_input("주인공 이름",
                                                                    st.session_state.story_data['protagonist_name'])
    st.session_state.story_data['protagonist_age'] = st.text_input("나이/직업",
                                                                   st.session_state.story_data['protagonist_age'])
with col2:
    st.session_state.story_data['story_genre'] = st.selectbox("장르", ["판타지", "로맨스", "SF", "일상", "스릴러"],
                                                              index=["판타지", "로맨스", "SF", "일상", "스릴러"].index(
                                                                  st.session_state.story_data['story_genre']))
    st.session_state.story_data['story_background'] = st.text_input("배경",
                                                                    st.session_state.story_data['story_background'])

# --- 멀티 캐릭터 설정 ---
st.header("👥 멀티 캐릭터 설정")
st.caption("조연, 악당 등 캐릭터 정보를 입력하세요.")
st.session_state.secondary_characters = st.data_editor(
    st.session_state.secondary_characters,
    num_rows="dynamic",
    use_container_width=True,
    column_config={"역할": st.column_config.SelectboxColumn("역할", options=["조연", "악당", "도우미", "기타"], required=True)}
)

st.markdown("---")
st.session_state.story_data['start_prompt'] = st.text_area(
    "이야기 시작 문구 (프롬프트)",
    value=st.session_state.story_data['start_prompt'],
    height=100
)

# --- 고급 제어 (사이드바) ---
st.sidebar.markdown("---")
st.sidebar.header("🧠 문체 및 톤 제어")

st.session_state.story_data['emotional_tone'] = st.sidebar.slider(
    "감성 톤 슬라이더 (어두움 -5 ↔ 밝음 5)",
    min_value=-5, max_value=5, value=st.session_state.story_data['emotional_tone'], step=1,
    help="-5는 어둡고 비극적, 5는 밝고 희망적인 분위기를 유도합니다."
)
st.session_state.story_data['sentence_structure'] = st.sidebar.radio(
    "문장 구조/길이",
    ["짧고 간결 (대화체)", "보통 (서술적)", "길고 복잡 (문어체)"],
    index=["짧고 간결 (대화체)", "보통 (서술적)", "길고 복잡 (문어체)"].index(st.session_state.story_data['sentence_structure'])
)

st.session_state.story_data['narrative_style'] = st.sidebar.selectbox("서술 시점", ["3인칭 (과거형)", "1인칭 (현재형)"])
st.session_state.story_data['persona_keywords'] = st.sidebar.text_input("캐릭터 페르소나",
                                                                        st.session_state.story_data['persona_keywords'])
st.session_state.story_data['required_plot'] = st.sidebar.text_area("필수 플롯",
                                                                    st.session_state.story_data['required_plot'])

# --- 파라미터 ---
st.sidebar.header("⚙️ 생성 파라미터")
temperature = st.sidebar.slider("창의성 (Temperature)", 0.1, 1.5, 0.9, help="값이 높을수록 예측 불가능하고 창의적인 문장을 생성합니다.")
# 💡 max_length -> max_output_tokens으로 명확히 변경
max_output_tokens = st.sidebar.slider("새로 생성할 길이 (토큰)", 50, 500, 200, help="프롬프트 제외, 새로 생성할 단어(토큰)의 최대 개수입니다.")
top_k = st.sidebar.slider("Top K", 1, 100, 50, help="매번 다음 단어를 고를 때 확률이 높은 K개의 후보군 안에서만 선택합니다.")
top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.95, help="확률을 누적하여 P%가 되는 최소한의 후보군 안에서만 선택합니다.")
penalty = st.sidebar.slider("반복 억제", 1.0, 2.0, 1.2, help="이전에 나왔던 단어가 다시 나올 확률을 낮춥니다.")

# ================= 버튼 영역 =================
st.markdown("---")
st.header("🎬 스토리 생성")

col_gen1, col_gen2 = st.columns(2)

# 1. 일반 생성
if col_gen1.button("✨ 일반 생성 시작", use_container_width=True):
    final_prompt = build_final_prompt(st.session_state.story_data['start_prompt'])
    bad_words = st.session_state.story_data['bad_words'].split(',')

    with st.spinner("스토리를 생성하는 중..."):
        res = generate_story(final_prompt, max_output_tokens, temperature, top_k, top_p, penalty, bad_words)
        clean_res = res.replace(final_prompt, "").strip()
        st.session_state.story_data['story_output'] = clean_res

# 2. 5단 구성
if col_gen2.button("🗺️ 5단 구성 모드", use_container_width=True):
    final_prompt = build_final_prompt(st.session_state.story_data['start_prompt'])
    bad_words = st.session_state.story_data['bad_words'].split(',')

    with st.spinner("5단 구성을 생성하는 중..."):
        # 5단 구성은 내부적으로 150 토큰으로 고정하여 호출
        res = generate_five_stages_mode(final_prompt, max_output_tokens, temperature, top_k, top_p, penalty, bad_words)
        st.session_state.story_data['story_output'] = res

st.markdown("---")

# --- 🔄 연속 생성 및 확장 제어 기능 ---
st.header("🔄 연속 생성 및 장면 확장 제어")
col_ext1, col_ext2 = st.columns(2)

next_scene_length = col_ext1.slider("다음 장면 생성 길이 (토큰)", 50, 400, 150)

trigger_event = col_ext2.selectbox(
    "다음 장면 유도 이벤트",
    ["선택 안 함", "새로운 캐릭터 등장", "주인공에게 위기 부여", "이전 사건 복선 회수", "극적인 반전"],
)

# 상담/대화 입력
col_dia1, col_dia2 = st.columns(2)
with col_dia1:
    dialogue_char = st.text_input("대화할 상대 캐릭터 이름:", placeholder="예: 이수호 또는 강태오")
with col_dia2:
    consult_q = st.text_input("고민 상담 내용:", placeholder="주인공이 위기에서 어떻게 탈출할까요?")

if st.button("💬 대화/상담 요청", use_container_width=True):
    if dialogue_char and st.session_state.story_data['story_output']:
        # 대화 모드
        dialogue_instruction = f"이전 이야기: {st.session_state.story_data['story_output'][-200:].strip()} \n\n"
        dialogue_instruction += f"캐릭터 '{dialogue_char}'와의 대화 장면을 이어서 작성해 줘."

        bad_words = st.session_state.story_data['bad_words'].split(',')

        with st.spinner(f"'{dialogue_char}'와의 대화 생성 중..."):
            dialogue_res = generate_story(dialogue_instruction, next_scene_length, temperature, top_k, top_p, penalty,
                                          bad_words)

            st.session_state.story_data['story_output'] += "\n\n" + dialogue_res.replace(dialogue_instruction,
                                                                                         "").strip()

    elif consult_q:
        # 상담 모드
        con_prompt = f"당신은 스토리 작가입니다. 장르:{st.session_state.story_data['story_genre']}. 현재 스토리:{st.session_state.story_data['story_output'][-200:].strip()}. 고민:{consult_q}. 조언을 해주세요."
        bad_words = st.session_state.story_data['bad_words'].split(',')
        with st.spinner("상담 분석 중..."):
            advice = generate_story(con_prompt, 300, 0.7, 50, 0.95, 1.2, bad_words)
            clean_advice = advice.replace(con_prompt, "").strip()
            st.success(f"🤖 AI 조언: {clean_advice}")

    else:
        st.warning("대화할 캐릭터 이름과 현재 대사, 혹은 상담 내용을 입력하세요.")

# ================= 결과 출력 및 편집 =================
if st.session_state.story_data['story_output']:
    st.markdown("---")
    st.subheader("📝 결과 편집")

    edited = st.text_area(
        "내용을 수정하세요:",
        value=st.session_state.story_data['story_output'],
        height=300
    )
    st.session_state.story_data['story_output'] = edited

    col_e1, col_e2, col_e3 = st.columns([1, 1, 1])

    # 이어서 생성 버튼
    if col_e1.button("➡️ 수정된 내용으로 이어서 생성"):
        sentences = edited.split('.')
        last_context = ".".join(sentences[-4:-1]).strip()

        event_prompt = ""
        if trigger_event != "선택 안 함":
            event_prompt = f" [{trigger_event}] 사건을 포함하여 이어서 전개해줘."
        else:
            event_prompt = " 이어서 다음 사건을 전개해줘."

        st.session_state.story_data['start_prompt'] = f"이전 내용: {last_context}...{event_prompt}"
        st.experimental_rerun()

    # 저장
    csv_data = pd.DataFrame([st.session_state.story_data]).to_csv(index=False, encoding='utf-8-sig')
    col_e2.download_button("💾 설정 및 스토리 CSV 저장", csv_data, "my_story_config.csv", "text/csv")

    txt_data = st.session_state.story_data['story_output']
    col_e3.download_button("📜 스토리 내용 TXT 저장", txt_data, "my_story_content.txt", "text/plain")

st.markdown("---")
st.caption("KoGPT2 모델 기반의 고급 스토리텔링 도구입니다.")