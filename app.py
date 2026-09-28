import datetime
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# Configuração da página em modo wide
st.set_page_config(
    page_title="Painel Pro - Futebol Global IA (Auto)",
    page_icon="⚽",
    layout="wide",
)

# Estilização Moderna Dark Neon
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    h1, h2, h3 { color: #00ffcc !important; font-family: 'Inter', sans-serif; }
    .stMetric { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); padding: 15px; border-radius: 12px; border: 1px solid #374151; box-shadow: 0 4px 6px -1px rgba(0, 255, 204, 0.1); }
    .card-ia { background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 20px; border-radius: 12px; border: 2px solid #34d399; color: white; margin-bottom: 20px; box-shadow: 0 0 15px rgba(52, 211, 153, 0.4); }
    .card-palpite { background-color: #1f2937; padding: 15px; border-radius: 10px; border-left: 6px solid #00ffcc; margin-bottom: 10px; color: #f3f4f6; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Painel Pro - Futebol Global & Brasil IA (Auto)")
st.markdown(
    "<p style='color: #9ca3af;'>Inteligência estatística automatizada e"
    " previsões por Distribuição de Poisson</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


@st.cache_data(
    ttl=3600
)  # Atualiza os dados automaticamente a cada 1 hora em cache web
def carregar_dados_inteligentes():
  try:
    # Tenta carregar a base atualizada
    df = pd.read_csv("dados_brasileirao.csv")
    return df
  except Exception:
    # Fallback seguro integrado para garantir estabilidade máxima
    return pd.DataFrame([
        {
            "liga": "Brasileirão Série A",
            "time": "Flamengo",
            "gols_pro_casa": 1.8,
            "gols_contra_casa": 0.7,
            "gols_pro_fora": 1.4,
            "gols_contra_fora": 1.0,
            "escanteios_pro": 6.2,
            "cartoes_pro": 2.1,
        },
        {
            "liga": "Brasileirão Série A",
            "time": "Palmeiras",
            "gols_pro_casa": 1.9,
            "gols_contra_casa": 0.6,
            "gols_pro_fora": 1.5,
            "gols_contra_fora": 0.9,
            "escanteios_pro": 5.9,
            "cartoes_pro": 2.3,
        },
        {
            "liga": "Premier League (Inglaterra)",
            "time": "Manchester City",
            "gols_pro_casa": 2.5,
            "gols_contra_casa": 0.6,
            "gols_pro_fora": 2.1,
            "gols_contra_fora": 0.8,
            "escanteios_pro": 7.1,
            "cartoes_pro": 1.5,
        },
        {
            "liga": "Premier League (Inglaterra)",
            "time": "Arsenal",
            "gols_pro_casa": 2.2,
            "gols_contra_casa": 0.7,
            "gols_pro_fora": 1.8,
            "gols_contra_fora": 0.9,
            "escanteios_pro": 6.5,
            "cartoes_pro": 1.8,
        },
    ])


df = carregar_dados_inteligentes()

if "liga" not in df.columns:
  df["liga"] = "Brasileirão Série A"

st.sidebar.header("🌍 Competição")
lista_ligas = sorted(df["liga"].unique())
liga_selecionada = st.sidebar.selectbox("🏆 Competição", lista_ligas)

df_liga = df[df["liga"] == liga_selecionada]
lista_times = sorted(df_liga["time"].unique())

if "historico_palpites" not in st.session_state:
  st.session_state["historico_palpites"] = []

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Equipes")
time_casa = st.sidebar.selectbox("🏟️ Mandante", lista_times, index=0)

times_visitante = [t for t in lista_times if t != time_casa]
if not times_visitante:
  times_visitante = lista_times

time_visitante = st.sidebar.selectbox("✈️ Visitante", times_visitante, index=0)

if time_casa == time_visitante:
  st.sidebar.warning("⚠️ Escolha equipas diferentes!")
else:
  dados_c = df_liga[df_liga["time"] == time_casa].iloc[0]
  dados_v = df_liga[df_liga["time"] == time_visitante].iloc[0]

  g_pro_c, g_contra_c = dados_c["gols_pro_casa"], dados_c["gols_contra_casa"]
  g_pro_f, g_contra_f = dados_v["gols_pro_fora"], dados_v["gols_contra_fora"]
  cantos_c, cartoes_c = dados_c["escanteios_pro"], dados_c["cartoes_pro"]
  cantos_f, cartoes_f = dados_v["escanteios_pro"], dados_v["cartoes_pro"]

  exp_gols_casa = (g_pro_c + g_contra_f) / 2
  exp_gols_visitante = (g_pro_f + g_contra_c) / 2
  total_gols_esperados = exp_gols_casa + exp_gols_visitante

  max_gols = 7
  matriz_prob = np.outer(
      poisson.pmf(np.arange(max_gols), exp_gols_casa),
      poisson.pmf(np.arange(max_gols), exp_gols_visitante),
  )

  prob_casa = np.sum(np.tril(matriz_prob, -1)) * 100
  prob_empate = np.sum(np.diag(matriz_prob)) * 100
  prob_visitante = np.sum(np.triu(matriz_prob, 1)) * 100
  prob_btts = np.sum(matriz_prob[1:, 1:]) * 100

  prob_gols_total = np.zeros(max_gols * 2)
  for i in range(max_gols):
    for j in range(max_gols):
      prob_gols_total[i + j] += matriz_prob[i, j] * 100

  p_menos_2 = np.sum(prob_gols_total[0:2])
  p_mais_2 = 100.0 - p_menos_2

  if prob_btts > 58:
    sugestao_ia = f"Ambas Marcam (BTTS: Sim) ({prob_btts:.1f}%)"
  elif p_mais_2 > 60:
    sugestao_ia = f"Mais de 1.5 / 2 Gols ({p_mais_2:.1f}%)"
  elif prob_casa > 55:
    sugestao_ia = f"Vitória de {time_casa} ({prob_casa:.1f}%)"
  else:
    sugestao_ia = (
        f"Vitória / Empata Anula: {time_visitante} ({prob_visitante:.1f}%)"
    )

  aba1, aba2, aba3 = st.tabs(
      ["🤖 Palpite IA", "📊 Probabilidades", "💾 Histórico"]
  )

  with aba1:
    st.subheader(f"Confronto: **{time_casa} vs {time_visitante}**")
    st.markdown(
        f"""
            <div class="card-ia">
                <h3 style="margin: 0; color: #ffffff !important;">🤖 Palpite de Maior Probabilidade (IA)</h3>
                <p style="font-size: 18px; font-weight: bold; margin: 10px 0 0 0;">👉 {sugestao_ia}</p>
                <p style="font-size: 13px; margin: 5px 0 0 0; opacity: 0.9;">Média estimada de golos: <b>{total_gols_esperados:.2f}</b></p>
            </div>
        """,
        unsafe_allow_html=True,
    )

  with aba2:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Vitória {time_casa}", f"{prob_casa:.1f}%")
    c2.metric("Empate", f"{prob_empate:.1f}%")
    c3.metric(f"Vitória {time_visitante}", f"{prob_visitante:.1f}%")

    if st.button("💾 Guardar Palpite", type="primary"):
      registro = {
          "data": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
          "liga": liga_selecionada,
          "partida": f"{time_casa} x {time_visitante}",
          "palpite_ia": sugestao_ia,
          "media": round(total_gols_esperados, 2),
      }
      st.session_state["historico_palpites"].append(registro)
      st.success("Guardado com sucesso!")

  with aba3:
    if len(st.session_state["historico_palpites"]) == 0:
      st.info("Nenhum palpite guardado.")
    else:
      for p in st.session_state["historico_palpites"]:
        st.markdown(
            f"""
                <div class="card-palpite">
                    <b>[{p['data']}] {p['partida']}</b><br>
                    • IA: <b>{p['palpite_ia']}</b>
                </div>
            """,
            unsafe_allow_html=True,
        )