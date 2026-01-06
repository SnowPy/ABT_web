import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# ==========================================
# 0. 全局配置 & CSS
# ==========================================
st.set_page_config(page_title="熊猫实验中心 V3.8 (修复版)", layout="wide")

st.markdown("""
<style>
    .metric-card { padding: 20px; border-radius: 10px; border: 1px solid #eee; text-align: center; margin-bottom: 10px; transition: transform 0.2s; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .card-pos { background-color: rgba(46, 204, 113, 0.1); border-top: 4px solid #2ECC71; }
    .card-neg { background-color: rgba(231, 76, 60, 0.1); border-top: 4px solid #E74C3C; }
    .card-neu { background-color: #f8f9fa; border-top: 4px solid #bdc3c7; color: #7f8c8d; }
    .m-title { font-size: 0.9em; font-weight: bold; color: #555; text-transform: uppercase; letter-spacing: 1px; }
    .m-val { font-size: 1.8em; font-weight: 800; margin: 10px 0; color: #333; }
    .m-lift { font-weight: bold; font-size: 1.1em; }
    .m-sig { font-size: 0.8em; margin-top: 5px; font-style: italic; opacity: 0.8; }
    .text-pos { color: #27ae60; }
    .text-neg { color: #c0392b; }
    .text-neu { color: #7f8c8d; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心工具函数
# ==========================================
TEMPLATE_DIR = "saved_templates"
if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)

def save_template(module_name, template_name, data):
    filename = f"{module_name}_{template_name}.json"
    filepath = os.path.join(TEMPLATE_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return filepath

def load_template(module_name, template_name):
    filename = f"{module_name}_{template_name}.json"
    filepath = os.path.join(TEMPLATE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_template_list(module_name):
    files = [f for f in os.listdir(TEMPLATE_DIR) if f.startswith(f"{module_name}_") and f.endswith(".json")]
    return [f.replace(f"{module_name}_", "").replace(".json", "") for f in files]

# --- 状态初始化助手 (关键修复) ---
def init_state(key, default_value):
    """如果 key 不在 session_state 中，则初始化它"""
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- 算法引擎 ---
def get_stats_pack(val_a, n_a, std_a, val_b, n_b, std_b, metric_type='mean', is_two_sided=True):
    base_a = val_a / n_a if n_a > 0 else 0
    base_b = val_b / n_b if n_b > 0 else 0
    lift = (base_b - base_a) / base_a if base_a != 0 else 0
    p_val = 1.0
    if n_a > 1 and n_b > 1: 
        if metric_type == 'mean':
            if std_a > 0 or std_b > 0:
                se = np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
                if se > 0:
                    t_stat = (base_b - base_a) / se
                    df = n_a + n_b - 2
                    p_val = stats.t.sf(np.abs(t_stat), df) * 2 if is_two_sided else stats.t.sf(t_stat, df)
        elif metric_type == 'prop':
            pool_p = (val_a + val_b) / (n_a + n_b)
            se = np.sqrt(pool_p * (1 - pool_p) * (1/n_a + 1/n_b))
            if se > 0:
                z = (base_b - base_a) / se
                p_val = stats.norm.sf(abs(z)) * 2 if is_two_sided else stats.norm.sf(z)
    state = 'neu'
    if p_val < 0.05: state = 'pos' if lift > 0 else 'neg'
    return {'base_b': base_b, 'lift': lift, 'p': p_val, 'state': state}

def calc_sample_size_prop(p_baseline, mde_rel, alpha=0.05, power=0.8, is_two_sided=True):
    p1 = p_baseline
    p2 = p1 * (1 + mde_rel)
    delta = abs(p2 - p1)
    if delta == 0: return 0
    z_alpha = stats.norm.ppf(1 - alpha/2) if is_two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    pooled_var = p1*(1-p1) + p2*(1-p2)
    return int(np.ceil(((z_alpha + z_beta)**2 * pooled_var) / (delta**2)))

def calc_sample_size_mean(mu, sigma, mde_rel, alpha=0.05, power=0.8, is_two_sided=True):
    mu1 = mu
    mu2 = mu1 * (1 + mde_rel)
    delta = abs(mu2 - mu1)
    if delta == 0: return 0
    z_alpha = stats.norm.ppf(1 - alpha/2) if is_two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    return int(np.ceil(((z_alpha + z_beta)**2 * 2 * (sigma**2)) / (delta**2)))

# ==========================================
# 2. 侧边栏：核心模式导航
# ==========================================
st.sidebar.title("🐼 熊猫实验中心")
app_mode = st.sidebar.radio("功能模块", ["📊 实验结果归因 (Analysis)", "🧪 实验设计 (Design)"], index=0)
st.sidebar.markdown("---")

# ==========================================
# 3. 模块 A: 实验结果归因 (Analysis)
# ==========================================
if app_mode == "📊 实验结果归因 (Analysis)":
    
    # --- 1. 预初始化 Session State (Analysis) ---
    # 关键步骤：先定义好所有默认值，后续 input 不再传 value 参数
    init_state('g_uv_a', 10000)
    init_state('g_rev_a', 50000.0)
    init_state('g_pay_a', 500)
    init_state('g_std_a', 120.0)
    init_state('g_uv_b', 10000)
    init_state('g_rev_b', 55000.0)
    init_state('g_pay_b', 480)
    init_state('g_std_b', 140.0)
    init_state('is_two_sided_mode_an', True)
    
    if 'node_list' not in st.session_state:
        st.session_state.node_list = [
            {"name": "详情页", "ua": 8000, "pa": 500, "ra": 50000, "sa": 120, "ub": 8200, "pb": 480, "rb": 55000, "sb": 140},
            {"name": "加购",   "ua": 4000, "pa": 500, "ra": 50000, "sa": 120, "ub": 3000, "pb": 480, "rb": 55000, "sb": 140},
        ]
    
    # --- 2. 模板管理 (Analysis) ---
    st.sidebar.subheader("💾 模板管理")
    an_templates = get_template_list("analysis")
    
    col_t1, col_t2 = st.sidebar.columns([2, 1])
    selected_template = col_t1.selectbox("选择模板加载", ["-- 请选择 --"] + an_templates, label_visibility="collapsed")
    if col_t2.button("加载"):
        if selected_template != "-- 请选择 --":
            data = load_template("analysis", selected_template)
            if data:
                # 严格类型转换，防止 JSON 数字类型错乱导致组件报错
                for k, v in data.items():
                    if k == 'node_list':
                        st.session_state[k] = v
                    elif isinstance(st.session_state.get(k), int):
                        st.session_state[k] = int(v) # 强制转回 int
                    elif isinstance(st.session_state.get(k), float):
                        st.session_state[k] = float(v) # 强制转回 float
                    else:
                        st.session_state[k] = v
                
                # 同步动态节点的 Key
                if 'node_list' in data:
                    for i, node in enumerate(data['node_list']):
                        st.session_state[f"name_{i}"] = node['name']
                        st.session_state[f"ua_{i}"] = int(node['ua'])
                        st.session_state[f"pa_{i}"] = int(node['pa'])
                        st.session_state[f"ra_{i}"] = float(node['ra'])
                        st.session_state[f"sa_{i}"] = float(node['sa'])
                        st.session_state[f"ub_{i}"] = int(node['ub'])
                        st.session_state[f"pb_{i}"] = int(node['pb'])
                        st.session_state[f"rb_{i}"] = float(node['rb'])
                        st.session_state[f"sb_{i}"] = float(node['sb'])

                st.success("加载成功")
                st.rerun()

    with st.sidebar.expander("保存当前配置为模板"):
        new_template_name = st.text_input("模板名称", placeholder="例如: 首页改版漏斗_V1")
        if st.button("保存模板"):
            if new_template_name:
                save_data = {
                    'g_uv_a': st.session_state['g_uv_a'],
                    'g_rev_a': st.session_state['g_rev_a'],
                    'g_pay_a': st.session_state['g_pay_a'],
                    'g_std_a': st.session_state['g_std_a'],
                    'g_uv_b': st.session_state['g_uv_b'],
                    'g_rev_b': st.session_state['g_rev_b'],
                    'g_pay_b': st.session_state['g_pay_b'],
                    'g_std_b': st.session_state['g_std_b'],
                    'node_list': st.session_state.get('node_list', []),
                    'is_two_sided_mode_an': st.session_state.get('is_two_sided_mode_an', True)
                }
                save_template("analysis", new_template_name, save_data)
                st.success(f"模板 '{new_template_name}' 已保存")
                st.rerun()
    st.sidebar.markdown("---")

    # --- 3. 输入区 (移除 value 参数，依赖 session_state) ---
    st.sidebar.subheader("1. 统计设置")
    
    test_mode_idx = 0 if st.session_state.is_two_sided_mode_an else 1
    test_mode = st.sidebar.radio("假设类型", ("双侧检验 (Two-sided)", "单侧检验 (One-sided B>A)"), index=test_mode_idx)
    st.session_state.is_two_sided_mode_an = (test_mode == "双侧检验 (Two-sided)")
    is_two_sided_mode = st.session_state.is_two_sided_mode_an
    
    with st.sidebar.expander("2. 全局基础数据", expanded=True):
        col_ga, col_gb = st.columns(2)
        with col_ga:
            st.markdown("🅰️ **对照组**")
            # 【修复点】: 移除了 value=...，Streamlit 会自动使用 session_state[key] 的值和类型
            st.number_input("总 UV", min_value=0, key='g_uv_a')
            st.number_input("总营收", min_value=0.0, key='g_rev_a')
            st.number_input("总付费", min_value=0, key='g_pay_a')
            st.number_input("Std Dev", min_value=0.0, key='g_std_a')
        with col_gb:
            st.markdown("🅱️ **实验组**")
            st.number_input("总 UV", min_value=0, key='g_uv_b')
            st.number_input("总营收", min_value=0.0, key='g_rev_b')
            st.number_input("总付费", min_value=0, key='g_pay_b')
            st.number_input("Std Dev", min_value=0.0, key='g_std_b')

    st.sidebar.subheader("3. 节点链路数据")
    c_add, c_reset = st.sidebar.columns([1,1])
    if c_add.button("➕ 新增节点"):
        st.session_state.node_list.append({"name": "新节点", "ua": 0, "pa": 0, "ra": 0.0, "sa": 0.0, "ub": 0, "pb": 0, "rb": 0.0, "sb": 0.0})
        st.rerun() # 强制刷新以初始化新节点的 Key
    if c_reset.button("🔄 重置数据"):
        st.session_state.node_list = []
        st.rerun()

    for i, node in enumerate(st.session_state.node_list):
        # 预初始化动态节点的 key
        init_state(f"name_{i}", node['name'])
        init_state(f"ua_{i}", node['ua'])
        init_state(f"pa_{i}", node['pa'])
        init_state(f"ra_{i}", node['ra'])
        init_state(f"sa_{i}", node['sa'])
        init_state(f"ub_{i}", node['ub'])
        init_state(f"pb_{i}", node['pb'])
        init_state(f"rb_{i}", node['rb'])
        init_state(f"sb_{i}", node['sb'])
        
        with st.sidebar.expander(f"📍 {st.session_state[f'name_{i}']}", expanded=False):
            st.text_input("节点名称", key=f"name_{i}")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("UV (A)", min_value=0, key=f"ua_{i}")
                st.number_input("付费数 (A)", min_value=0, key=f"pa_{i}")
                st.number_input("营收 (A)", min_value=0.0, key=f"ra_{i}")
                st.number_input("Std (A)", min_value=0.0, key=f"sa_{i}")
            with c2:
                st.number_input("UV (B)", min_value=0, key=f"ub_{i}")
                st.number_input("付费数 (B)", min_value=0, key=f"pb_{i}")
                st.number_input("营收 (B)", min_value=0.0, key=f"rb_{i}")
                st.number_input("Std (B)", min_value=0.0, key=f"sb_{i}")
            
            # 将 Widget 的值同步回 node_list (用于计算和保存)
            st.session_state.node_list[i]['name'] = st.session_state[f"name_{i}"]
            st.session_state.node_list[i]['ua'] = st.session_state[f"ua_{i}"]
            st.session_state.node_list[i]['pa'] = st.session_state[f"pa_{i}"]
            st.session_state.node_list[i]['ra'] = st.session_state[f"ra_{i}"]
            st.session_state.node_list[i]['sa'] = st.session_state[f"sa_{i}"]
            st.session_state.node_list[i]['ub'] = st.session_state[f"ub_{i}"]
            st.session_state.node_list[i]['pb'] = st.session_state[f"pb_{i}"]
            st.session_state.node_list[i]['rb'] = st.session_state[f"rb_{i}"]
            st.session_state.node_list[i]['sb'] = st.session_state[f"sb_{i}"]

    # 获取 widget 更新后的值用于计算
    g_uv_a = st.session_state['g_uv_a']
    g_rev_a = st.session_state['g_rev_a']
    g_pay_a = st.session_state['g_pay_a']
    g_std_a = st.session_state['g_std_a']
    g_uv_b = st.session_state['g_uv_b']
    g_rev_b = st.session_state['g_rev_b']
    g_pay_b = st.session_state['g_pay_b']
    g_std_b = st.session_state['g_std_b']
    
    node_df = pd.DataFrame([{
        "节点名称": n['name'],
        "UV_A": n['ua'], "付费数_A": n['pa'], "营收_A": n['ra'], "Std_A": n['sa'],
        "UV_B": n['ub'], "付费数_B": n['pb'], "营收_B": n['rb'], "Std_B": n['sb']
    } for n in st.session_state.node_list])

    # --- Analysis Logic ---
    st.title("🐼 实验结果归因 (Analysis)")
    
    g_arpu = get_stats_pack(g_rev_a, g_uv_a, g_std_a, g_rev_b, g_uv_b, g_std_b, 'mean', is_two_sided_mode)
    g_cvr = get_stats_pack(g_pay_a, g_uv_a, 0, g_pay_b, g_uv_b, 0, 'prop', is_two_sided_mode)
    g_asp = get_stats_pack(g_rev_a, g_pay_a, g_std_a, g_rev_b, g_pay_b, g_std_b, 'mean', is_two_sided_mode)

    def render_html_card(title, res, is_pct=False):
        icon = "▲" if res['lift'] > 0 else "▼"
        val_fmt = f"{res['base_b']*100:.2f}%" if is_pct else f"¥{res['base_b']:.2f}"
        if res['state'] == 'pos': sig_text = "显著提升 (Significant)"
        elif res['state'] == 'neg': sig_text = "显著下降 (Significant)"
        else: sig_text = "差异不显著 (Not Sig)"
        return f"""
        <div class="metric-card card-{res['state']}">
            <div class="m-title">{title}</div>
            <div class="m-val">{val_fmt}</div>
            <div class="m-lift text-{res['state']}">{icon} {abs(res['lift']*100):.2f}%</div>
            <div class="m-sig">P = {res['p']:.3f}<br>{sig_text}</div>
        </div>
        """

    st.subheader("1. 全局动力拆解")
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(render_html_card("全局 ARPU", g_arpu, False), unsafe_allow_html=True)
    with col2: st.markdown(render_html_card("全局 CVR", g_cvr, True), unsafe_allow_html=True)
    with col3: st.markdown(render_html_card("全局 ASP", g_asp, False), unsafe_allow_html=True)

    st.divider()
    st.subheader("2. 全链路诊断图谱")
    if node_df.empty:
        st.info("👈 请在侧边栏添加节点数据")
    else:
        plot_data = []
        insights = []
        for _, row in node_df.iterrows():
            name = row['节点名称']
            r_arpu = get_stats_pack(row['营收_A'], row['UV_A'], row['Std_A'], row['营收_B'], row['UV_B'], row['Std_B'], 'mean', is_two_sided_mode)
            r_cvr = get_stats_pack(row['付费数_A'], row['UV_A'], 0, row['付费数_B'], row['UV_B'], 0, 'prop', is_two_sided_mode)
            r_asp = get_stats_pack(row['营收_A'], row['付费数_A'], row['Std_A'], row['营收_B'], row['付费数_B'], row['Std_B'], 'mean', is_two_sided_mode)
            for m, res in zip(['Node-ARPU', 'Node-CVR', 'Node-ASP'], [r_arpu, r_cvr, r_asp]):
                plot_data.append({'node': name, 'metric': m, 'lift': res['lift'], 'state': res['state'], 'p': res['p']})
            if r_cvr['state']=='neg' and r_asp['state']=='pos':
                insights.append(f"⚖️ **[{name}]** 出现“洗用户”现象：转化率跌 {r_cvr['lift']*100:.1f}% 但客单价涨 {r_asp['lift']*100:.1f}%。")

        df_plot = pd.DataFrame(plot_data)
        color_map = {'pos': '#2ECC71', 'neg': '#E74C3C', 'neu': '#95a5a6'}
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Node-ARPU", "Node-CVR", "Node-ASP"), shared_yaxes=True)
        metrics = ['Node-ARPU', 'Node-CVR', 'Node-ASP']

        for i, m in enumerate(metrics):
            d = df_plot[df_plot['metric'] == m]
            for _, r in d.iterrows():
                fig.add_shape(type="line", x0=0, y0=r['node'], x1=r['lift'], y1=r['node'], line=dict(color=color_map[r['state']], width=2), row=1, col=i+1)
            marker_symbols = ['circle' if r['p'] < 0.05 else 'circle-open' for _, r in d.iterrows()]
            marker_sizes = [14 if r['p'] < 0.05 else 10 for _, r in d.iterrows()]
            fig.add_trace(go.Scatter(
                x=d['lift'], y=d['node'], mode='markers+text',
                marker=dict(color=[color_map[x] for x in d['state']], symbol=marker_symbols, size=marker_sizes, line=dict(width=2, color=[color_map[x] for x in d['state']])),
                text=[f"{x*100:.1f}%" for x in d['lift']], textposition="top center", textfont=dict(size=10, color="#555"), showlegend=False
            ), row=1, col=i+1)
            fig.add_vline(x=0, line_dash="dash", line_color="#ccc", row=1, col=i+1)
        
        fig.update_layout(height=400, showlegend=False, yaxis={'autorange': "reversed"}, margin=dict(t=50))
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴/🟢 实心点 = P<0.05 (Significant); ⚪ 空心点 = P≥0.05")
        if insights: st.info("\n".join(insights))

# ==========================================
# 4. 模块 B: 实验设计 (Sample Size)
# ==========================================
elif app_mode == "🧪 实验设计 (Design)":
    
    # --- 1. 预初始化 (Design) ---
    init_state('alpha_input', 0.05)
    init_state('power_input', 0.80)
    init_state('daily_traffic', 1000)
    init_state('base_cvr', 0.05)
    init_state('mde_cvr_raw', 10.0)
    init_state('base_mean', 10.0)
    init_state('base_std_design', 20.0)
    init_state('mde_mean_raw', 5.0)
    init_state('is_two_sided_design_mode', True)

    # --- 2. 模板管理 (Design) ---
    st.sidebar.subheader("💾 模板管理")
    de_templates = get_template_list("design")
    col_t1, col_t2 = st.sidebar.columns([2, 1])
    sel_temp_de = col_t1.selectbox("选择模板加载", ["-- 请选择 --"] + de_templates, label_visibility="collapsed", key='sel_temp_de')
    if col_t2.button("加载", key='btn_load_de'):
        if sel_temp_de != "-- 请选择 --":
            data = load_template("design", sel_temp_de)
            if data:
                for k, v in data.items(): st.session_state[k] = v
                st.success("加载成功")
                st.rerun()

    with st.sidebar.expander("保存当前配置为模板"):
        new_template_name_de = st.text_input("模板名称", placeholder="例如: 95置信度_高敏度", key='new_t_de')
        if st.button("保存模板", key='btn_save_de'):
            if new_template_name_de:
                save_data = {
                    'alpha_input': st.session_state['alpha_input'],
                    'power_input': st.session_state['power_input'],
                    'daily_traffic': st.session_state['daily_traffic'],
                    'base_cvr': st.session_state['base_cvr'],
                    'mde_cvr_raw': st.session_state['mde_cvr_raw'],
                    'base_mean': st.session_state['base_mean'],
                    'base_std_design': st.session_state['base_std_design'],
                    'mde_mean_raw': st.session_state['mde_mean_raw'],
                    'is_two_sided_design_mode': st.session_state.get('is_two_sided_design_mode', True)
                }
                save_template("design", new_template_name_de, save_data)
                st.success(f"模板 '{new_template_name_de}' 已保存")
                st.rerun()
    st.sidebar.markdown("---")

    st.title("🧪 样本量估算 (Sample Size Calculator)")
    
    # --- Input Configuration (移除 value 参数) ---
    with st.sidebar:
        st.subheader("1. 统计参数设置")
        alpha_input = st.number_input("显著性水平 (α)", step=0.01, key='alpha_input')
        power_input = st.number_input("统计功效 (1-β)", step=0.01, key='power_input')
        
        idx_de = 0 if st.session_state.is_two_sided_design_mode else 1
        test_mode_design = st.radio("检验假设", ("双侧检验 (Two-sided)", "单侧检验 (One-sided)"), index=idx_de, key='radio_design_mode')
        st.session_state.is_two_sided_design_mode = (test_mode_design == "双侧检验 (Two-sided)")
        is_two_sided_design = st.session_state.is_two_sided_design_mode
        
        st.markdown("---")
        st.subheader("2. 流量估算 (可选)")
        daily_traffic = st.number_input("预估单组日均流量", key='daily_traffic')

    tab1, tab2 = st.tabs(["📊 转化率类指标 (CVR)", "💰 数值类指标 (ARPU/ASP)"])
    
    # TAB 1: CVR
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            base_cvr = st.number_input("当前基准转化率 (Baseline CVR)", format="%.4f", key='base_cvr')
        with c2:
            mde_cvr_raw = st.number_input("预期相对提升 (MDE %)", key='mde_cvr_raw')
            mde_cvr = mde_cvr_raw / 100
            
        if base_cvr > 0 and mde_cvr != 0:
            req_n = calc_sample_size_prop(base_cvr, mde_cvr, alpha_input, power_input, is_two_sided_design)
            days = np.ceil(req_n / daily_traffic) if daily_traffic > 0 else 0
            
            st.success(f"### 🎯 单组所需样本量: {req_n:,} 人")
            if days > 0:
                st.info(f"⏳ 基于日均 {daily_traffic} 人，预估需要跑 **{int(days)}** 天")
            
            st.markdown("#### 📉 灵敏度分析: MDE vs 样本量")
            mde_range = np.linspace(0.01, 0.50, 50) 
            n_list = [calc_sample_size_prop(base_cvr, m, alpha_input, power_input, is_two_sided_design) for m in mde_range]
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(x=mde_range*100, y=n_list, mode='lines', name='Sample Size', line=dict(color='#2980b9', width=3)))
            fig_sens.add_trace(go.Scatter(x=[mde_cvr*100], y=[req_n], mode='markers', marker=dict(color='red', size=12), name='当前设定'))
            fig_sens.update_layout(xaxis_title="预期相对提升 (MDE %)", yaxis_title="单组样本量", height=300, margin=dict(t=20, l=20, r=20, b=20))
            st.plotly_chart(fig_sens, use_container_width=True)

    # TAB 2: ARPU
    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            base_mean = st.number_input("当前均值 (Baseline Mean)", key='base_mean')
        with c2:
            base_std = st.number_input("当前标准差 (Std Dev)", key='base_std_design')
        with c3:
            mde_mean_raw = st.number_input("预期相对提升 (MDE %)", key='mde_mean_raw')
            mde_mean_pct = mde_mean_raw / 100
            
        if base_mean > 0 and mde_mean_pct != 0:
            req_n_mean = calc_sample_size_mean(base_mean, base_std, mde_mean_pct, alpha_input, power_input, is_two_sided_design)
            days_mean = np.ceil(req_n_mean / daily_traffic) if daily_traffic > 0 else 0
            
            st.success(f"### 🎯 单组所需样本量: {req_n_mean:,} 人")
            if days_mean > 0:
                st.info(f"⏳ 基于日均 {daily_traffic} 人，预估需要跑 **{int(days_mean)}** 天")
            
            st.markdown("#### 📉 灵敏度分析: MDE vs 样本量")
            mde_range_mean = np.linspace(0.01, 0.20, 50) 
            n_list_mean = [calc_sample_size_mean(base_mean, base_std, m, alpha_input, power_input, is_two_sided_design) for m in mde_range_mean]
            fig_sens_m = go.Figure()
            fig_sens_m.add_trace(go.Scatter(x=mde_range_mean*100, y=n_list_mean, mode='lines', name='Sample Size', line=dict(color='#8e44ad', width=3)))
            fig_sens_m.add_trace(go.Scatter(x=[mde_mean_pct*100], y=[req_n_mean], mode='markers', marker=dict(color='red', size=12), name='当前设定'))
            fig_sens_m.update_layout(xaxis_title="预期相对提升 (MDE %)", yaxis_title="单组样本量", height=300, margin=dict(t=20, l=20, r=20, b=20))
            st.plotly_chart(fig_sens_m, use_container_width=True)