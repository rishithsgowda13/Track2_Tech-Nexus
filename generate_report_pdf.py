from fpdf import FPDF
import datetime

class NexusReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(0, 150, 0) # Nexus Green
        self.cell(0, 10, 'NEXUS VISION: SYSTEM ARCHITECTURE REPORT', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.datetime.now().strftime("%Y-%m-%d")}', 0, 0, 'C')

def create_pdf():
    pdf = NexusReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Executive Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, '1. System Overview', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, 'Nexus Vision is a high-performance situational awareness and autonomous navigation system designed for off-road environments. It uses a Dual-Stream Agentic Fusion architecture to perceive both terrain semantics and object instances simultaneously.')
    pdf.ln(5)

    # 2. Workflow
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Core Workflow Breakdown', 0, 1, 'L')
    
    workflow = [
        ("Step I: Perception Stack", "Terrain Segmentation processes features from a DINOv2 (Vision Transformer) backbone. In parallel, YOLOv8 identifies dynamic obstacles."),
        ("Step II: Agentic Fusion", "The system fuses the semantic terrain map with the object detection stream for redundant safety."),
        ("Step III: Dynamic Pathfinding", "Calculates an optimal steering vector by identifying the largest safe gap in the environment."),
        ("Step IV: Visualization", "Provides a 4-panel AI-brain view for real-time situational awareness.")
    ]
    
    for title, desc in workflow:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 7, f'  - {title}', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, f'    {desc}')
        pdf.ln(2)

    # 3. Technical Specs (Table)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. Technical Specifications', 0, 1, 'L')
    
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(40, 7, 'Component', 1, 0, 'C', 1)
    pdf.cell(60, 7, 'Technology', 1, 0, 'C', 1)
    pdf.cell(90, 7, 'Role', 1, 1, 'C', 1)
    
    specs = [
        ("Backbone", "DINOv2 (ViT-Small)", "Feature extraction"),
        ("Segmentation", "ConvNeXt Head", "Terrain classification"),
        ("Detection", "YOLOv8-Nano", "Obstacle awareness"),
        ("Navigation", "Dynamic Gap-Finder", "Steering logic")
    ]
    
    pdf.set_font('Arial', '', 9)
    for comp, tech, role in specs:
        pdf.cell(40, 7, comp, 1)
        pdf.cell(60, 7, tech, 1)
        pdf.cell(90, 7, role, 1, 1)

    # Output
    pdf.output("SYSTEM_ARCHITECTURE_REPORT.pdf")
    print("✅ PDF Report Generated: SYSTEM_ARCHITECTURE_REPORT.pdf")

if __name__ == "__main__":
    create_pdf()
