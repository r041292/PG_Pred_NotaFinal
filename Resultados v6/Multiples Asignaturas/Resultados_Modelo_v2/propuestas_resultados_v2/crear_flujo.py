from graphviz import Digraph

# Create the flowchart
dot = Digraph(comment='Flujo y Resultados', format='png')
dot.attr(rankdir='TB')  # Top to Bottom layout
dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='9', penwidth='2')
dot.attr('edge', fontname='Arial', fontsize='9', penwidth='2', color='#4a6fa5')

# Color palette
color_normal = '#bbdefb'      # Light blue for most boxes
color_special = '#42a5f5'     # Darker blue for 5A and 5B
fontcolor_normal = '#01579b'  # Dark blue text for light background
fontcolor_special = 'white'   # White text for darker background

# Nodes
dot.node('input', 'CSV DE ENTRADA\nResultados_Modelo.csv\nfilas=21607', shape='cylinder', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.15')

dot.node('step1', '[1] NORMALIZAR\nPrediccion_final_XGB\nconvertir coma a punto\ncrear columna num', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.15')

dot.node('step2', '[2] CLASIFICAR\nEstado XGBOOST\nRetiro: pred < 0\nPerdida: 0 <= pred < 3\nAprobacion: pred >= 3', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.15')

dot.node('step3', '[3] MARCAR ASIGNATURAS A IGNORAR\nfilas_ignoradas=1181\nfilas_utiles=20426', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.2')

dot.node('step4', '[4] CALCULAR REPITENCIA\nasignaturas_utiles=172\nasignaturas_con_repitencia=123\nnrc_unicos=794', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.2')

dot.node('step5a', '[5A] REPITENCIA ALTA\numbral = media + 1*desv\nasignaturas_altas=11', fillcolor=color_special, fontcolor=fontcolor_special, margin='0.15')

dot.node('step5b', '[5B] NRC SOBRESALIENTES\numbral = media + 1*desv\nnrc_sobresalientes=94', fillcolor=color_special, fontcolor=fontcolor_special, margin='0.15')

dot.node('step6', '[6] INTEGRAR REGLAS Y PRIORIZAR TOP N\ntop_n=100\nasignaturas_seleccionadas=100', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.2')

dot.node('step7', '[7] GRUPOS FINALES\nApoyo en NRC: 50\nApoyo en Asignatura: 44\nApoyo en Ambas: 6', shape='box', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.15')

# Edges
dot.edge('input', 'step1')
dot.edge('step1', 'step2')
dot.edge('step2', 'step3')
dot.edge('step3', 'step4')
dot.edge('step4', 'step5a')
dot.edge('step4', 'step5b')
dot.edge('step5a', 'step6')
dot.edge('step5b', 'step6')
dot.edge('step6', 'step7')

# Render - Standard version
dot.render('propuesta_asignaturas_nrc_flujo', cleanup=True)
print('Diagrama generado exitosamente: propuesta_asignaturas_nrc_flujo.png')

# Render - Transparent background version
dot_transparent = Digraph(comment='Flujo y Resultados', format='png')
dot_transparent.attr(rankdir='TB')
dot_transparent.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='9', penwidth='2')
dot_transparent.attr('edge', fontname='Arial', fontsize='9', penwidth='2', color='#4a6fa5')
dot_transparent.attr(bgcolor='transparent')

for node_id, node_attrs in dot.body[0].items() if hasattr(dot.body[0], 'items') else []:
    pass

dot_transparent.node('input', 'CSV DE ENTRADA\nResultados_Modelo.csv\nfilas=21607', shape='cylinder', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.15')
dot_transparent.node('step1', '[1] NORMALIZAR\nPrediccion_final_XGB\nconvertir coma a punto\ncrear columna num', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.15')
dot_transparent.node('step2', '[2] CLASIFICAR\nEstado XGBOOST\nRetiro: pred < 0\nPerdida: 0 <= pred < 3\nAprobacion: pred >= 3', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.15')
dot_transparent.node('step3', '[3] MARCAR ASIGNATURAS A IGNORAR\nfilas_ignoradas=1181\nfilas_utiles=20426', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.2')
dot_transparent.node('step4', '[4] CALCULAR REPITENCIA\nasignaturas_utiles=172\nasignaturas_con_repitencia=123\nnrc_unicos=794', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.2')
dot_transparent.node('step5a', '[5A] REPITENCIA ALTA\numbral = media + 1*desv\nasignaturas_altas=11', fillcolor=color_special, fontcolor=fontcolor_special, margin='0.15')
dot_transparent.node('step5b', '[5B] NRC SOBRESALIENTES\numbral = media + 1*desv\nnrc_sobresalientes=94', fillcolor=color_special, fontcolor=fontcolor_special, margin='0.15')
dot_transparent.node('step6', '[6] INTEGRAR REGLAS Y PRIORIZAR TOP N\ntop_n=100\nasignaturas_seleccionadas=100', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.2')
dot_transparent.node('step7', '[7] GRUPOS FINALES\nApoyo en NRC: 50\nApoyo en Asignatura: 44\nApoyo en Ambas: 6', shape='box', fillcolor=color_normal, fontcolor=fontcolor_normal, margin='0.15')

dot_transparent.edge('input', 'step1')
dot_transparent.edge('step1', 'step2')
dot_transparent.edge('step2', 'step3')
dot_transparent.edge('step3', 'step4')
dot_transparent.edge('step4', 'step5a')
dot_transparent.edge('step4', 'step5b')
dot_transparent.edge('step5a', 'step6')
dot_transparent.edge('step5b', 'step6')
dot_transparent.edge('step6', 'step7')

dot_transparent.render('propuesta_asignaturas_nrc_flujo_transparente', cleanup=True)
print('Diagrama transparente generado exitosamente: propuesta_asignaturas_nrc_flujo_transparente.png')
