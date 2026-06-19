from graphviz import Digraph

# Create the flowchart - Methodology section
dot = Digraph(comment='Metodologia', format='png')
dot.attr(rankdir='TB')
dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10', penwidth='2')
dot.attr('edge', fontname='Arial', fontsize='9', penwidth='2', color='#666666')

# Gray color palette
colors = {
    'light': '#f5f5f5',      # Very light gray
    'medium_light': '#e0e0e0',  # Light gray
    'medium': '#bdbdbd',     # Medium gray
    'medium_dark': '#9e9e9e', # Medium-dark gray
    'dark': '#757575'        # Dark gray
}
fontcolor = '#212121'  # Very dark gray for text

# Nodes - Simplified methodology flow
dot.node('step1', '1. Normalización, Clasificación y Depuración\n• Convertir predicción a Retiro, Pérdida y Aprobación\n• Excluir asignaturas no relevantes', fillcolor=colors['light'], fontcolor=fontcolor, margin='0.15')

dot.node('step2', '2. Cálculo de Repitencia\n• Repitencia = Retiro + Pérdida\n• Nivel asignatura\n• Nivel grupo (NRC)', fillcolor=colors['medium_light'], fontcolor=fontcolor, margin='0.15')

dot.node('step3a', '3a. Priorización de Asignaturas\n• Top 100 por volumen de repitencia\n• Umbral: media + 1*desv', fillcolor=colors['medium'], fontcolor=fontcolor, margin='0.15')

dot.node('step3b', '3b. Identificación de Grupos Críticos\n• Repitencia > media + 1*desv\n• Dentro de cada asignatura', fillcolor=colors['medium'], fontcolor=fontcolor, margin='0.15')

dot.node('step4', '4. Estrategia de Intervención\n• Apoyo en NRC\n• Apoyo en Asignatura\n• Apoyo en Ambas', shape='folder', fillcolor=colors['dark'], fontcolor='white', margin='0.15')

# Edges
dot.edge('step1', 'step2')
dot.edge('step2', 'step3a')
dot.edge('step2', 'step3b')
dot.edge('step3a', 'step4')
dot.edge('step3b', 'step4')

# Render - Standard version
dot.render('metodologia_flujo', cleanup=True)
print('Diagrama generado exitosamente: metodologia_flujo.png')

# Render - Transparent background version
dot_transparent = Digraph(comment='Metodologia', format='png')
dot_transparent.attr(rankdir='TB')
dot_transparent.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10', penwidth='2')
dot_transparent.attr('edge', fontname='Arial', fontsize='9', penwidth='2', color='#666666')
dot_transparent.attr(bgcolor='transparent')

dot_transparent.node('step1', '1. Normalización, Clasificación y Depuración\n• Convertir predicción a Retiro, Pérdida y Aprobación\n• Excluir asignaturas no relevantes', fillcolor=colors['light'], fontcolor=fontcolor, margin='0.15')
dot_transparent.node('step2', '2. Cálculo de Repitencia\n• Repitencia = Retiro + Pérdida\n• Nivel asignatura\n• Nivel grupo (NRC)', fillcolor=colors['medium_light'], fontcolor=fontcolor, margin='0.15')
dot_transparent.node('step3a', '3a. Priorización de Asignaturas\n• Top 100 por volumen de repitencia\n• Umbral: media + 1*desv', fillcolor=colors['medium'], fontcolor=fontcolor, margin='0.15')
dot_transparent.node('step3b', '3b. Identificación de Grupos Críticos\n• Repitencia > media + 1*desv\n• Dentro de cada asignatura', fillcolor=colors['medium'], fontcolor=fontcolor, margin='0.15')
dot_transparent.node('step4', '4. Estrategia de Intervención\n• Apoyo en NRC\n• Apoyo en Asignatura\n• Apoyo en Ambas', shape='folder', fillcolor=colors['dark'], fontcolor='white', margin='0.15')

dot_transparent.edge('step1', 'step2')
dot_transparent.edge('step2', 'step3a')
dot_transparent.edge('step2', 'step3b')
dot_transparent.edge('step3a', 'step4')
dot_transparent.edge('step3b', 'step4')

dot_transparent.render('metodologia_flujo_transparente', cleanup=True)
print('Diagrama transparente generado exitosamente: metodologia_flujo_transparente.png')
