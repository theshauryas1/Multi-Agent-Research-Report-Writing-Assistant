"""
Export Utilities for generating Markdown and PDF reports.
"""

import os
from datetime import datetime
from typing import List, Dict, Optional

from config import OUTPUT_DIR


def ensure_output_dir():
    """Ensure the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def export_to_markdown(
    title: str,
    content: str,
    sources: Optional[List[Dict[str, str]]] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Export report content to a Markdown file.
    
    Args:
        title: Report title
        content: Main report content (already in markdown format)
        sources: Optional list of source dictionaries with 'title' and 'url'
        filename: Optional custom filename (auto-generated if not provided)
    
    Returns:
        Path to the exported file
    """
    ensure_output_dir()
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = title.lower().replace(" ", "_")[:30]
        filename = f"report_{safe_title}_{timestamp}.md"
    
    # Build the markdown content
    md_content = f"# {title}\n\n"
    md_content += f"*Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}*\n\n"
    md_content += "---\n\n"
    md_content += content
    
    # Add sources section if provided
    if sources:
        md_content += "\n\n---\n\n## References\n\n"
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            if url:
                md_content += f"{i}. [{title}]({url})\n"
            else:
                md_content += f"{i}. {title}\n"
    
    # Write to file
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return filepath


def export_to_pdf(
    title: str,
    content: str,
    sources: Optional[List[Dict[str, str]]] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Export report content to a PDF file.
    
    Args:
        title: Report title
        content: Main report content
        sources: Optional list of source dictionaries
        filename: Optional custom filename
    
    Returns:
        Path to the exported file
    """
    from fpdf import FPDF
    
    ensure_output_dir()
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = title.lower().replace(" ", "_")[:30]
        filename = f"report_{safe_title}_{timestamp}.pdf"
    
    # Create PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 24)
    pdf.multi_cell(0, 12, title, align="C")
    pdf.ln(5)
    
    # Date
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C")
    pdf.ln(15)
    
    # Content - parse markdown and render
    pdf.set_font("Helvetica", "", 11)
    
    # Simple markdown parsing
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        
        if not line:
            pdf.ln(5)
            continue
        
        # Handle headers
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 8, line[4:])
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.ln(5)
            pdf.multi_cell(0, 9, line[3:])
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.ln(8)
            pdf.multi_cell(0, 10, line[2:])
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("- "):
            # Bullet points
            pdf.cell(10)  # Indent
            pdf.multi_cell(0, 7, f"• {line[2:]}")
        elif line.startswith("**") and line.endswith("**"):
            # Bold text
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 7, line[2:-2])
            pdf.set_font("Helvetica", "", 11)
        else:
            # Normal paragraph
            pdf.multi_cell(0, 7, line)
    
    # Add sources if provided
    if sources:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "References", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "", 10)
        for i, source in enumerate(sources, 1):
            source_title = source.get("title", "Untitled")
            url = source.get("url", "")
            text = f"{i}. {source_title}"
            if url:
                text += f" - {url}"
            pdf.multi_cell(0, 6, text)
            pdf.ln(2)
    
    # Save PDF
    filepath = os.path.join(OUTPUT_DIR, filename)
    pdf.output(filepath)
    
    return filepath


def export_to_html(
    title: str,
    content: str,
    sources: Optional[List[Dict[str, str]]] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Export report content to an HTML file.
    
    Args:
        title: Report title
        content: Main report content (markdown)
        sources: Optional list of source dictionaries
        filename: Optional custom filename
    
    Returns:
        Path to the exported file
    """
    ensure_output_dir()
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = title.lower().replace(" ", "_")[:30]
        filename = f"report_{safe_title}_{timestamp}.html"
    
    # Convert basic markdown to HTML
    html_content = _markdown_to_html(content)
    
    # Build HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.6;
            color: #333;
            background: #fafafa;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
        }}
        h3 {{
            color: #27ae60;
        }}
        .meta {{
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 30px;
        }}
        .content {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .references {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
        }}
        .references a {{
            color: #3498db;
            text-decoration: none;
        }}
        .references a:hover {{
            text-decoration: underline;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="meta">Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
    <div class="content">
        {html_content}
    </div>
"""
    
    # Add sources section
    if sources:
        html += """    <div class="references">
        <h2>References</h2>
        <ol>
"""
        for source in sources:
            source_title = source.get("title", "Untitled")
            url = source.get("url", "")
            if url:
                html += f'            <li><a href="{url}" target="_blank">{source_title}</a></li>\n'
            else:
                html += f'            <li>{source_title}</li>\n'
        html += """        </ol>
    </div>
"""
    
    html += """</body>
</html>"""
    
    # Write to file
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    
    return filepath


def _markdown_to_html(md_text: str) -> str:
    """Simple markdown to HTML conversion."""
    import re
    
    html = md_text
    
    # Headers
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Italic
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Bullet points
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # Wrap consecutive <li> in <ul>
    html = re.sub(r'(<li>.*?</li>\n)+', lambda m: f'<ul>\n{m.group()}</ul>\n', html)
    
    # Paragraphs (lines not starting with HTML tags)
    lines = html.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('<'):
            result.append(f'<p>{stripped}</p>')
        else:
            result.append(line)
    
    return '\n'.join(result)


if __name__ == "__main__":
    # Test exports
    test_title = "Test Report"
    test_content = """## Introduction

This is a test report to demonstrate the export functionality.

## Main Content

Here is some **bold** and *italic* text.

- Point one
- Point two
- Point three

### Subsection

More detailed information here.

## Conclusion

The export system works correctly.
"""
    
    test_sources = [
        {"title": "Source One", "url": "https://example.com/1"},
        {"title": "Source Two", "url": "https://example.com/2"},
    ]
    
    print("Testing exports...")
    
    md_path = export_to_markdown(test_title, test_content, test_sources)
    print(f"✓ Markdown exported to: {md_path}")
    
    pdf_path = export_to_pdf(test_title, test_content, test_sources)
    print(f"✓ PDF exported to: {pdf_path}")
    
    html_path = export_to_html(test_title, test_content, test_sources)
    print(f"✓ HTML exported to: {html_path}")
