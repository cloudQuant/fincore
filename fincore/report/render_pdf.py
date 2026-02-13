"""PDF report renderer — generates HTML then converts via Playwright.

Uses Playwright (headless Chromium) to render the HTML report to PDF,
then optionally adds bookmarks via PyPDF2.
"""

from __future__ import annotations

import os
import tempfile


def generate_pdf(
    returns,
    benchmark_rets,
    positions,
    transactions,
    trades,
    title,
    output,
    rolling_window,
):
    """生成 PDF 报告：先生成 HTML，再用 Playwright 渲染为 PDF。"""
    from fincore.report.render_html import generate_html

    # 1) 生成临时 HTML 文件
    out_dir = os.path.dirname(os.path.abspath(output)) or "."
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, dir=out_dir) as tmp:
        tmp_html = tmp.name

    generate_html(
        returns,
        benchmark_rets=benchmark_rets,
        positions=positions,
        transactions=transactions,
        trades=trades,
        title=title,
        output=tmp_html,
        rolling_window=rolling_window,
    )

    # 2) 用 Playwright 将 HTML 渲染为 PDF
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "生成 PDF 需要 playwright 库，请执行:\n  pip install playwright && python -m playwright install chromium"
        )

    # 临时 PDF 路径（后续添加书签后写入最终 output）
    tmp_pdf = output + ".tmp.pdf"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        page.goto(f"file://{os.path.abspath(tmp_html)}", wait_until="networkidle", timeout=60000)

        # 智能等待：检测所有 ECharts 实例渲染完毕
        page.evaluate("""() => {
            return new Promise((resolve) => {
                let attempts = 0;
                const check = () => {
                    attempts++;
                    const containers = document.querySelectorAll('[id^="c-"]');
                    let allReady = true;
                    containers.forEach(el => {
                        const canvas = el.querySelector('canvas');
                        if (!canvas) allReady = false;
                    });
                    if (allReady || attempts > 30) resolve();
                    else setTimeout(check, 200);
                };
                setTimeout(check, 500);
            });
        }""")
        # 额外等待确保图表动画完成
        page.wait_for_timeout(1500)

        # 收集各节的标题和页面位置（用于生成书签）
        section_info = page.evaluate("""() => {
            const sections = document.querySelectorAll('.sec');
            const results = [];
            sections.forEach(sec => {
                const titleEl = sec.querySelector('.sec-title');
                if (titleEl) {
                    const rect = sec.getBoundingClientRect();
                    results.push({
                        id: sec.id,
                        title: titleEl.textContent.trim(),
                        top: rect.top + window.scrollY
                    });
                }
            });
            // 文档总高度
            const totalHeight = document.documentElement.scrollHeight;
            return { sections: results, totalHeight: totalHeight };
        }""")

        page.pdf(
            path=tmp_pdf,
            format="A4",
            print_background=True,
            margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
        )
        browser.close()

    # 3) 清理临时 HTML
    try:
        os.remove(tmp_html)
    except OSError:
        pass

    # 4) 添加 PDF 书签（可点击目录）
    _add_pdf_bookmarks(tmp_pdf, output, section_info, title)

    # 清理临时 PDF
    try:
        os.remove(tmp_pdf)
    except OSError:
        pass

    return output


def _add_pdf_bookmarks(input_pdf, output_pdf, section_info, report_title):
    """给 PDF 添加可点击的书签/大纲目录。"""
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        # 无 PyPDF2 时直接复制文件
        import shutil

        shutil.copy2(input_pdf, output_pdf)
        return

    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    # 复制所有页面
    for page in reader.pages:
        writer.add_page(page)

    # 计算文档总高度到页面的映射
    total_pages = len(reader.pages)
    if total_pages == 0:
        with open(output_pdf, "wb") as f:
            writer.write(f)
        return

    sections = section_info.get("sections", [])

    # 每个 A4 页面的 CSS 像素高度（约 1123px at 96dpi for A4 minus margins）
    # Playwright uses 96dpi; A4 = 297mm ≈ 1123px, minus margins (12mm*2 = ~91px)
    page_css_height = 1123 - 91  # ≈ 1032px per page content area

    # 添加根书签
    writer.add_outline_item(report_title, 0)

    for sec in sections:
        sec_top = sec["top"]
        sec_title = sec["title"]

        # 估算此节在第几页
        est_page = int(sec_top / page_css_height) if page_css_height > 0 else 0
        est_page = min(est_page, total_pages - 1)

        writer.add_outline_item(sec_title, est_page)

    with open(output_pdf, "wb") as f:
        writer.write(f)
