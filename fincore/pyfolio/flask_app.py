from flask import Flask, render_template
import pandas as pd
import os
import shutil
import glob


# 初始化 Flask 应用
app = Flask(__name__,
            template_folder="templates",
            static_folder="static")

# 静态文件和模板文件路径
data_root = os.path.dirname(__file__)  # 当前文件目录
static_path = os.path.join(data_root, "static")
template_path = os.path.join(data_root, "templates")
# 数据文件路径
files = {
    "performance_df": "strategy_performance_Out-of-sample months.xlsx",
    "stress_events_df": "strategy_performance_Stress Events.xlsx",
    "long_position_df": "strategy_performance_Top 10 long positions of all time.xlsx",
    "short_position_df": "strategy_performance_Top 10 short positions of all time.xlsx",
    "total_position_df": "strategy_performance_Top 10 positions of all time.xlsx",
    "total_months_df": "strategy_performance_Total months.xlsx",
    "worst_drawdown_df": "strategy_performance_Worst drawdown periods.xlsx",
}


# 获取静态文件中的图片路径
def get_static_images():
    return [
        f"image/returns_tear_sheet.png",
        f"image/interesting_times_tear_sheet.png",
        f"image/position_tear_sheet.png",
    ]


# 加载数据函数
def load_excel_data(file_name, index_col=0):
    file_path = os.path.join(static_path, file_name)
    if os.path.exists(file_path):
        try:
            return pd.read_excel(file_path, index_col=index_col)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
    return pd.DataFrame()  # 返回空 DataFrame 作为备用


# 加载所有数据表
performance_df = load_excel_data(files["performance_df"])
performance_df.index.name = "performance analysis"
performance_df.reset_index(inplace=True)

stress_events_df = load_excel_data(files["stress_events_df"])
stress_events_df.index.name = 'Stress Events'
stress_events_df.reset_index(inplace=True)

long_position_df = load_excel_data(files["long_position_df"])
long_position_df.index.name = 'Top 10 long positions of all time'
long_position_df.reset_index(inplace=True)

short_position_df = load_excel_data(files["short_position_df"])
short_position_df.index.name = 'Top 10 short positions of all time'
short_position_df.reset_index(inplace=True)

worst_drawdown_df = load_excel_data(files["worst_drawdown_df"])
worst_drawdown_df.index.name = 'Worst drawdown periods'
worst_drawdown_df.reset_index(inplace=True)

total_position_df = load_excel_data(files["total_position_df"])
total_position_df.index.name = 'Top 10 positions of all time'
total_position_df.reset_index(inplace=True)

total_months_df = load_excel_data(files["total_months_df"])
total_months_df.index.name = 'total months'
total_months_df.reset_index(inplace=True)

@app.route("/", methods=("GET",))
def index():
    # 渲染模板
    target_file = "index.html"
    static_images = get_static_images()
    return render_template(
        target_file,
        name="Pyfolio - 策略绩效分析",
        static_images=static_images,  # 传递图片列表到模板
        performance_tables=[performance_df.to_html(classes='data', header=True, index=False)],
        stress_events_tables=[stress_events_df.to_html(classes='data', header=True, index=False)],
        long_position_tables=[long_position_df.to_html(classes='data', header=True, index=False)],
        short_position_tables=[short_position_df.to_html(classes='data', header=True, index=False)],
        worst_drawdown_tables=[worst_drawdown_df.to_html(classes='data', header=True, index=False)],
        total_position_tables=[total_position_df.to_html(classes='data', header=True, index=False)],
        total_months_tables=[total_months_df.to_html(classes='data', header=True, index=False)],
        )


if __name__ == "__main__":
    # 启动应用
    app.run(host="0.0.0.0", port=2025, debug=True)
