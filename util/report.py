import os
import base64

def export_report(OUTPUT):
    print('[LOG] Exporting HTML report')
    try:
        with open(OUTPUT, 'w', encoding="utf-8") as file:
            for fig in os.listdir('./output/temp_figures'):
                with open('./output/temp_figures/' + fig, 'rb') as svg_file:
                    svg_base64 = str(base64.b64encode(svg_file.read()),'utf-8')
                    file.write(f'<img src="data:image/svg+xml;base64,{svg_base64}" />\n')
                os.remove('./output/temp_figures/' + fig)
    
        print('[LOG] HTML report created')
    except:
        print('[ERROR] Failed to create HTML report')