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
st.set_page_config(page_title="熊猫实验中心 V3.9 (Pro)", layout="wide")

st.markdown("""
<style>
    .metric-card { 
        background-color: white;
        padding: 0px; 
        border-radius: 12px; 
        border: 1px solid #e0e0e0; 
        text-align: center; 
        margin-bottom: 15px; 
        transition: all 0.2s ease-in-out; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.04); 
        overflow: hidden;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); }
    
    /* 顶部状态条 */
    .card-pos { border-top: 6px solid #2ECC71; }
    .card-neg { border-top: 6px solid #E74C3C; }
    .card-neu { border-top: 6px solid #bdc3c7; }
    
    /* 内容区域 */
    .m-content { padding: 20px 15px 10px 15px; }
    .m-title { font-size: 0.85em; font-weight: 700; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;}
    .m-val { font-size: 1.8em; font-weight: 800; color: #2c3e50; line-height: 1.2; }
    .m-lift { font-weight: bold; font-size: 1.1em; margin: 8px 0; display: flex; justify-content: center; align-items: center; gap: 5px;}
    
    /* 底部详情条 */
    .m-footer { 
        background-color: #f8f9fa; 
        padding: 10px; 
        border-top: 1px solid #eee; 
        font-size: 0.75em; 
        color: #666;
        display: flex;
        justify-content: space-around;
    }
    .m-footer-item { display: flex; flex-direction: column; line-height: 1.3; }
    .mf-label { font-weight: 600; color: #aaa; font-size: 0.9em; }
    
    .text-pos { color: #27ae60; }
    .text-neg { color: #c0392b; }
    .text-neu { color: #95a5a6; }
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

def init_state(key, default_value):
    """如果 key 不在 session_state 中，则初始化它"""
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- 核心算法引擎 (升级版: 含 CI 与 Power) ---
def get_stats_pack(val_a, n_a, std_a, val_b, n_b, std_b, metric_type='mean', is_two_sided=True, alpha=0.05):
    """
    计算核心统计指标：Lift, P-value, CI (置信区间), Power (统计功效)
    """
    # 1. 基础计算
    base_a = val_a / n_a if n_a > 0 else 0
    base_b = val_b / n_b if n_b > 0 else 0
    diff = base_b - base_a
    lift = diff / base_a if base_a != 0 else 0
    
    p_val = 1.0
    se_diff = 0.0
    power_val = 0.0
    ci_low, ci_high = 0.0, 0.0
    
    if n_a > 1 and n_b > 1: 
        # 2. 标准误差 (SE) 计算
        if metric_type == 'mean':
            # Mean: Welch's t-test assumption (unequal variances)
            if std_a > 0 or std_b > 0:
                se_diff = np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
        elif metric_type == 'prop':
            # Prop: Pooled SE (often used for p-value) vs Unpooled SE (better for CI)
            # 这里为了区间估计的一致性，使用 Unpooled SE 计算 CI
            p_a = base_a
            p_b = base_b
            se_diff = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)

        # 3. 推断统计 (P-value, CI, Power)
        if se_diff > 0:
            # --- P-Value ---
            z_score = diff / se_diff
            if is_two_sided:
                p_val = stats.norm.sf(abs(z_score)) * 2
                crit_val = stats.norm.ppf(1 - alpha/2)
            else:
                p_val = stats.norm.sf(z_score)
                crit_val = stats.norm.ppf(1 - alpha)
            
            # --- Confidence Interval (Absolute & Relative) ---
            # Absolute CI
            ci_abs_low = diff - crit_val * se_diff
            ci_abs_high = diff + crit_val * se_diff
            
            # Relative CI (转化为相对于 base_a 的百分比)
            # 注意：这是 Delta Method 的简化近似，假设分母 base_a 相对稳定
            if base_a != 0:
                ci_low = ci_abs_low / base_a
                ci_high = ci_abs_high / base_a
            
            # --- Observed Power (Post-hoc) ---
            # 计算在当前样本量下，检测出"当前观测差异(diff)"的概率
            # Power = P(reject H0 | H1 is true)
            # Shift the distribution by z_score
            # approx power calculation
            power_val = stats.norm.cdf(abs(z_score) - crit_val)
            # 如果是单侧且 diff 为负，power 可能很小，这里取绝对值代表检测"差异"的能力
            
    # 4. 状态判定
    state = 'neu'
    if p_val < alpha: 
        state = 'pos' if lift > 0 else 'neg'
        
    return {
        'base_a': base_a,
        'base_b': base_b,
        'lift': lift,
        'p': p_val,
        'state': state,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'power': power_val,
        'se': se_diff
    }

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
app_mode = st.sidebar.radio("功能模块", 
    ["📊 实验结果归因 (Analysis)", "🧪 实验设计 (Design)", "🔮 贝叶斯推断 (Bayesian)"], 
    index=0
)
st.sidebar.markdown("---")

# ==========================================
# 3. 模块 A: 实验结果归因 (Analysis)
# ==========================================
if app_mode == "📊 实验结果归因 (Analysis)":
    
    # --- 1. 预初始化 Session State ---
    init_state('g_uv_a', 10000)
    init_state('g_rev_a', 50000.0)
    init_state('g_pay_a', 500)
    init_state('g_std_a', 120.0)
    init_state('g_uv_b', 10000)
    init_state('g_rev_b', 56000.0) # 稍微调高一点以便展示显著性
    init_state('g_pay_b', 520)
    init_state('g_std_b', 125.0)
    init_state('is_two_sided_mode_an', True)
    
    if 'node_list' not in st.session_state:
        st.session_state.node_list = [
            {"name": "详情页", "ua": 8000, "pa": 500, "ra": 50000, "sa": 120, "ub": 8200, "pb": 510, "rb": 56000, "sb": 125},
            {"name": "加购",   "ua": 4000, "pa": 500, "ra": 50000, "sa": 120, "ub": 4050, "pb": 480, "rb": 55000, "sb": 140},
        ]
    
    # --- 2. 模板管理 ---
    st.sidebar.subheader("💾 模板管理")
    an_templates = get_template_list("analysis")
    
    col_t1, col_t2 = st.sidebar.columns([2, 1])
    selected_template = col_t1.selectbox("选择模板加载", ["-- 请选择 --"] + an_templates, label_visibility="collapsed")
    if col_t2.button("加载"):
        if selected_template != "-- 请选择 --":
            data = load_template("analysis", selected_template)
            if data:
                for k, v in data.items():
                    if k == 'node_list':
                        st.session_state[k] = v
                    elif k in st.session_state:
                         # 简单的类型恢复
                        if isinstance(st.session_state[k], int): st.session_state[k] = int(v)
                        elif isinstance(st.session_state[k], float): st.session_state[k] = float(v)
                        else: st.session_state[k] = v
                
                # 同步动态节点 Key
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

    # --- 3. 输入区 ---
    st.sidebar.subheader("1. 统计设置")
    
    test_mode_idx = 0 if st.session_state.is_two_sided_mode_an else 1
    test_mode = st.sidebar.radio("假设类型", ("双侧检验 (Two-sided)", "单侧检验 (One-sided B>A)"), index=test_mode_idx)
    st.session_state.is_two_sided_mode_an = (test_mode == "双侧检验 (Two-sided)")
    is_two_sided_mode = st.session_state.is_two_sided_mode_an
    
    alpha_val = st.sidebar.number_input("显著性水平 (α)", value=0.05, step=0.01, min_value=0.01, max_value=0.20)

    with st.sidebar.expander("2. 全局基础数据", expanded=True):
        col_ga, col_gb = st.columns(2)
        with col_ga:
            st.markdown("🅰️ **对照组**")
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
        st.rerun()
    if c_reset.button("🔄 重置数据"):
        st.session_state.node_list = []
        st.rerun()

    for i, node in enumerate(st.session_state.node_list):
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
            
            st.session_state.node_list[i]['name'] = st.session_state[f"name_{i}"]
            st.session_state.node_list[i]['ua'] = st.session_state[f"ua_{i}"]
            st.session_state.node_list[i]['pa'] = st.session_state[f"pa_{i}"]
            st.session_state.node_list[i]['ra'] = st.session_state[f"ra_{i}"]
            st.session_state.node_list[i]['sa'] = st.session_state[f"sa_{i}"]
            st.session_state.node_list[i]['ub'] = st.session_state[f"ub_{i}"]
            st.session_state.node_list[i]['pb'] = st.session_state[f"pb_{i}"]
            st.session_state.node_list[i]['rb'] = st.session_state[f"rb_{i}"]
            st.session_state.node_list[i]['sb'] = st.session_state[f"sb_{i}"]

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
    
    # 计算核心指标 (ARPU, CVR, ASP)
    # ARPU: Mean metric
    g_arpu = get_stats_pack(g_rev_a, g_uv_a, g_std_a, g_rev_b, g_uv_b, g_std_b, 'mean', is_two_sided_mode, alpha_val)
    # CVR: Prop metric (std=0 placeholder)
    g_cvr = get_stats_pack(g_pay_a, g_uv_a, 0, g_pay_b, g_uv_b, 0, 'prop', is_two_sided_mode, alpha_val)
    # ASP: Mean metric
    g_asp = get_stats_pack(g_rev_a, g_pay_a, g_std_a, g_rev_b, g_pay_b, g_std_b, 'mean', is_two_sided_mode, alpha_val)

    def render_html_card(title, res, is_pct=False):
        icon = "▲" if res['lift'] > 0 else "▼"
        val_fmt = f"{res['base_b']*100:.2f}%" if is_pct else f"¥{res['base_b']:.2f}"
        
        # 显著性文本
        if res['state'] == 'pos': sig_text = "显著提升"
        elif res['state'] == 'neg': sig_text = "显著下降"
        else: sig_text = "差异不显著"
        
        # CI 格式化
        ci_low_fmt = f"{res['ci_low']*100:+.2f}%"
        ci_high_fmt = f"{res['ci_high']*100:+.2f}%"
        
        # Power 颜色：低于 0.8 显示警告色
        power_color = "#e67e22" if res['power'] < 0.8 else "#2c3e50"
        
        return f"""
        <div class="metric-card card-{res['state']}">
            <div class="m-content">
                <div class="m-title">{title}</div>
                <div class="m-val">{val_fmt}</div>
                <div class="m-lift text-{res['state']}">
                    {icon} {abs(res['lift']*100):.2f}%
                    <span style="font-size:0.7em; margin-left:5px; opacity:0.7"> (P={res['p']:.3f})</span>
                </div>
            </div>
            <div class="m-footer">
                <div class="m-footer-item">
                    <span class="mf-label">95% CI (Lift)</span>
                    <span style="font-family: monospace;">[{ci_low_fmt}, {ci_high_fmt}]</span>
                </div>
                <div style="border-left:1px solid #ddd;"></div>
                <div class="m-footer-item">
                    <span class="mf-label">Power (1-β)</span>
                    <span style="color:{power_color}; font-weight:bold;">{res['power']:.2f}</span>
                </div>
            </div>
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
            r_arpu = get_stats_pack(row['营收_A'], row['UV_A'], row['Std_A'], row['营收_B'], row['UV_B'], row['Std_B'], 'mean', is_two_sided_mode, alpha_val)
            r_cvr = get_stats_pack(row['付费数_A'], row['UV_A'], 0, row['付费数_B'], row['UV_B'], 0, 'prop', is_two_sided_mode, alpha_val)
            r_asp = get_stats_pack(row['营收_A'], row['付费数_A'], row['Std_A'], row['营收_B'], row['付费数_B'], row['Std_B'], 'mean', is_two_sided_mode, alpha_val)
            
            for m, res in zip(['Node-ARPU', 'Node-CVR', 'Node-ASP'], [r_arpu, r_cvr, r_asp]):
                plot_data.append({
                    'node': name, 'metric': m, 
                    'lift': res['lift'], 'state': res['state'], 'p': res['p'],
                    'ci_low': res['ci_low'], 'ci_high': res['ci_high'] # 绘图可用 error bar
                })
            
            if r_cvr['state']=='neg' and r_asp['state']=='pos':
                insights.append(f"⚖️ **[{name}]** 出现“洗用户”现象：转化率跌 {r_cvr['lift']*100:.1f}% 但客单价涨 {r_asp['lift']*100:.1f}%。")

        df_plot = pd.DataFrame(plot_data)
        color_map = {'pos': '#2ECC71', 'neg': '#E74C3C', 'neu': '#95a5a6'}
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Node-ARPU", "Node-CVR", "Node-ASP"), shared_yaxes=True)
        metrics = ['Node-ARPU', 'Node-CVR', 'Node-ASP']

        for i, m in enumerate(metrics):
            d = df_plot[df_plot['metric'] == m]
            
            # Error Bars (CI)
            # Plotly error_x 需要是绝对长度，不是坐标点
            # lift is x, ci_low is x_min. error_minus = lift - ci_low
            
            error_y_pos = [r['ci_high'] - r['lift'] for _, r in d.iterrows()]
            error_y_neg = [r['lift'] - r['ci_low'] for _, r in d.iterrows()]

            fig.add_trace(go.Scatter(
                x=d['lift'], y=d['node'], 
                mode='markers+text',
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=error_y_pos,
                    arrayminus=error_y_neg,
                    color='#ccc',
                    thickness=1,
                    width=3
                ),
                marker=dict(
                    color=[color_map[x] for x in d['state']], 
                    size=12,
                    line=dict(width=2, color='white')
                ),
                text=[f"{x*100:.1f}%" for x in d['lift']], 
                textposition="top center", 
                textfont=dict(size=10, color="#555"), 
                showlegend=False
            ), row=1, col=i+1)
            
            # 0 线
            fig.add_vline(x=0, line_dash="dash", line_color="#ddd", row=1, col=i+1)
        
        fig.update_layout(height=400, showlegend=False, yaxis={'autorange': "reversed"}, margin=dict(t=50))
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ℹ️ 图中灰色横线代表 95% 置信区间 (Confidence Interval)。若横线穿过 0 轴，则差异通常不显著。")
        
        if insights: st.info("\n".join(insights))

# ==========================================
# 4. 模块 B: 实验设计 (Sample Size)
# ==========================================
elif app_mode == "🧪 实验设计 (Design)":
    
    init_state('alpha_input', 0.05)
    init_state('power_input', 0.80)
    init_state('daily_traffic', 1000)
    init_state('base_cvr', 0.05)
    init_state('mde_cvr_raw', 10.0)
    init_state('base_mean', 10.0)
    init_state('base_std_design', 20.0)
    init_state('mde_mean_raw', 5.0)
    init_state('is_two_sided_design_mode', True)

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

# ==========================================
# 5. 模块 C: 贝叶斯推断 (Bayesian) - 升级版
# ==========================================
elif app_mode == "🔮 贝叶斯推断 (Bayesian)":
    
    # 状态初始化
    init_state('bay_vis_a', 500)
    init_state('bay_conv_a', 45)
    init_state('bay_rev_a', 1500.0) # 新增：收入
    
    init_state('bay_vis_b', 520)
    init_state('bay_conv_b', 62)
    init_state('bay_rev_b', 2100.0) # 新增：收入
    
    init_state('bay_prior_strength', '弱先验 (Weak)')
    init_state('bay_loss_threshold', 0.01) # 决策阈值

    st.title("🔮 贝叶斯推断 (Bayesian Inference)")
    st.markdown("""
    > **适用场景**: 小流量/小样本实验。  
    > **核心模型**: 
    > * **转化率 (CVR)**: Beta-Binomial 模型
    > * **ARPU (每用户收入)**: Hurdle Model (CVR $\\times$ ARPPU) + 蒙特卡罗模拟
    """)
    st.divider()

    # --- Sidebar Inputs ---
    with st.sidebar:
        st.subheader("1. 实验数据录入")
        
        col_ba, col_bb = st.columns(2)
        with col_ba:
            st.markdown("🅰️ **对照组 (A)**")
            st.number_input("样本量 (N)", min_value=1, key='bay_vis_a')
            st.number_input("转化人数 (K)", min_value=0, key='bay_conv_a')
            st.number_input("总收入 (Rev)", min_value=0.0, key='bay_rev_a')
        with col_bb:
            st.markdown("🅱️ **实验组 (B)**")
            st.number_input("样本量 (N)", min_value=1, key='bay_vis_b')
            st.number_input("转化人数 (K)", min_value=0, key='bay_conv_b')
            st.number_input("总收入 (Rev)", min_value=0.0, key='bay_rev_b')
            
        st.markdown("---")
        st.subheader("2. 参数配置")
        st.selectbox(
            "先验认知强度", 
            ["弱先验 (Weak)", "乐观先验 (Optimistic)"],
            key='bay_prior_strength',
            help="弱先验假设我们对A/B没有任何预设偏好 (Beta(1,1))"
        )
        st.number_input(
            "风险容忍阈值 (Loss Threshold)", 
            value=0.05, step=0.01, 
            key='bay_loss_threshold',
            help="如果B实际上比A差，你最多能容忍ARPU亏多少？"
        )
        st.info("💡 系统会自动进行 100,000 次蒙特卡罗模拟")

    # --- Calculation Engine (Monte Carlo) ---
    SIM_SIZE = 100000
    np.random.seed(42)
    
    # 获取输入
    N_a, K_a, R_a = st.session_state.bay_vis_a, st.session_state.bay_conv_a, st.session_state.bay_rev_a
    N_b, K_b, R_b = st.session_state.bay_vis_b, st.session_state.bay_conv_b, st.session_state.bay_rev_b
    
    # 1. CVR Posterior: Beta Distribution
    # Prior: Beta(1,1) for Weak
    prior_alpha, prior_beta = (1, 1) if st.session_state.bay_prior_strength.startswith("弱") else (5, 95)
    
    cvr_samples_a = stats.beta.rvs(prior_alpha + K_a, prior_beta + N_a - K_a, size=SIM_SIZE)
    cvr_samples_b = stats.beta.rvs(prior_alpha + K_b, prior_beta + N_b - K_b, size=SIM_SIZE)
    
    # 2. ARPPU Posterior: Inverse Gamma Logic (via Gamma Rate Parameter)
    # Model: Revenue ~ Exponential(lambda) => Mean = 1/lambda
    # Prior on lambda: Gamma(alpha, beta). Posterior lambda: Gamma(alpha+K, beta+Rev)
    # Weak Prior for Gamma: alpha=0.001, beta=0.001 (Uninformative)
    g_alpha, g_beta = 0.001, 0.001
    
    # Prevent division by zero if K=0
    if K_a > 0:
        lambda_a = stats.gamma.rvs(a=g_alpha + K_a, scale=1/(g_beta + R_a), size=SIM_SIZE)
        arppu_samples_a = 1 / lambda_a
    else:
        arppu_samples_a = np.zeros(SIM_SIZE)
        
    if K_b > 0:
        lambda_b = stats.gamma.rvs(a=g_alpha + K_b, scale=1/(g_beta + R_b), size=SIM_SIZE)
        arppu_samples_b = 1 / lambda_b
    else:
        arppu_samples_b = np.zeros(SIM_SIZE)

    # 3. ARPU (Hurdle Model Combination)
    # ARPU = CVR * ARPPU
    arpu_samples_a = cvr_samples_a * arppu_samples_a
    arpu_samples_b = cvr_samples_b * arppu_samples_b
    
    # --- Metrics Calculation ---
    # Probability B > A
    prob_b_win = (arpu_samples_b > arpu_samples_a).mean()
    
    # Expected Uplift (Relative)
    # Avoid div by zero
    safe_a = np.where(arpu_samples_a == 0, 1e-9, arpu_samples_a)
    uplift_dist = (arpu_samples_b - arpu_samples_a) / safe_a
    expected_uplift = np.median(uplift_dist) # Use median for stability in ratios
    
    # Expected Loss (Absolute Value Risk)
    # Loss = Mean of (A - B) where A > B
    loss_dist = np.maximum(arpu_samples_a - arpu_samples_b, 0)
    expected_loss = loss_dist.mean()
    
    # Current Observed ARPU
    obs_arpu_a = R_a / N_a if N_a > 0 else 0
    obs_arpu_b = R_b / N_b if N_b > 0 else 0

    # --- UI Rendering ---
    
    # Top Cards
    c1, c2, c3 = st.columns(3)
    
    # Card 1: Win Probability
    color_win = "normal"
    if prob_b_win > 0.95: label_win = "建议上线 (High Confidence)"
    elif prob_b_win > 0.8: label_win = "潜力观察 (Positive)"
    else: label_win = "风险较高/无差异"
    
    c1.metric(
        "🏆 B 胜出的概率 (Win Rate)", 
        f"{prob_b_win*100:.1f}%",
        delta=label_win,
        delta_color="normal" if prob_b_win > 0.5 else "inverse"
    )
    
    # Card 2: Expected Uplift
    c2.metric(
        "📈 预期 ARPU 提升 (Uplift)", 
        f"{expected_uplift*100:.2f}%",
        help="B 版本相对 A 版本 ARPU 的提升幅度中位数"
    )
    
    # Card 3: Expected Loss (Risk)
    is_safe = expected_loss < st.session_state.bay_loss_threshold
    c3.metric(
        "🛡️ 潜在风险 (Expected Loss)", 
        f"{expected_loss:.4f}",
        delta="风险可控" if is_safe else "风险过高",
        delta_color="normal" if is_safe else "inverse",
        help=f"如果选错了 B，平均每用户可能亏损的金额。阈值设定为 {st.session_state.bay_loss_threshold}"
    )

    # Tabs for Visuals
    t_main, t_cvr, t_arppu = st.tabs(["📊 ARPU 综合决策", "🔍 转化率分布", "💰 ARPPU 分布"])
    
    with t_main:
        st.markdown("#### ARPU 后验分布对比 (Posterior Density)")
        st.caption(f"基于 Hurdle Model ($CVR \\times ARPPU$) 的 10w 次模拟结果。观测 ARPU: A={obs_arpu_a:.2f}, B={obs_arpu_b:.2f}")
        
        fig_arpu = go.Figure()
        # Histogram/KDE approximation using density hist
        fig_arpu.add_trace(go.Histogram(x=arpu_samples_a, histnorm='probability density', name='Control A', marker_color='#95a5a6', opacity=0.6))
        fig_arpu.add_trace(go.Histogram(x=arpu_samples_b, histnorm='probability density', name='Variant B', marker_color='#2ECC71', opacity=0.6))
        
        fig_arpu.update_layout(
            xaxis_title="ARPU (Value)", yaxis_title="Density",
            barmode='overlay', height=350, margin=dict(t=10)
        )
        st.plotly_chart(fig_arpu, use_container_width=True)
        
        # Risk / Decision Logic Text
        if prob_b_win > 0.90 and is_safe:
            st.success(f"✅ **决策建议：发布 B 版本**。胜率高 ({prob_b_win*100:.1f}%) 且潜在损失 ({expected_loss:.4f}) 低于您的阈值。")
        elif prob_b_win < 0.10:
            st.error(f"🛑 **决策建议：放弃 B 版本**。A 版本胜率极高 ({100 - prob_b_win*100:.1f}%)。")
        else:
            st.warning(f"⚖️ **决策建议：继续实验**。虽然当前胜率为 {prob_b_win*100:.1f}%，但尚未达到 95% 确信度或风险 ({expected_loss:.4f}) 仍需关注。")

    with t_cvr:
        st.markdown("#### 转化率 (CVR) 独立分析")
        fig_cvr = go.Figure()
        x_cvr = np.linspace(0, max(cvr_samples_a.max(), cvr_samples_b.max())*1.1, 500)
        # Analytical Beta PDF for smooth lines
        y_a_cvr = stats.beta.pdf(x_cvr, prior_alpha + K_a, prior_beta + N_a - K_a)
        y_b_cvr = stats.beta.pdf(x_cvr, prior_alpha + K_b, prior_beta + N_b - K_b)
        
        fig_cvr.add_trace(go.Scatter(x=x_cvr, y=y_a_cvr, fill='tozeroy', name='CVR A', line=dict(color='#95a5a6')))
        fig_cvr.add_trace(go.Scatter(x=x_cvr, y=y_b_cvr, fill='tozeroy', name='CVR B', line=dict(color='#3498db')))
        fig_cvr.update_layout(height=350, xaxis_title="Conversion Rate")
        st.plotly_chart(fig_cvr, use_container_width=True)

    with t_arppu:
        st.markdown("#### 客单价 (ARPPU) 独立分析")
        st.caption("注：仅统计付费用户的平均价值分布 (Gamma-Inverse Model)")
        fig_arppu = go.Figure()
        # Use histograms for derived samples
        fig_arppu.add_trace(go.Histogram(x=arppu_samples_a, histnorm='probability density', name='ARPPU A', marker_color='#95a5a6', opacity=0.6))
        fig_arppu.add_trace(go.Histogram(x=arppu_samples_b, histnorm='probability density', name='ARPPU B', marker_color='#e67e22', opacity=0.6))
        fig_arppu.update_layout(height=350, barmode='overlay', xaxis_title="Average Revenue Per Paying User")
        st.plotly_chart(fig_arppu, use_container_width=True)