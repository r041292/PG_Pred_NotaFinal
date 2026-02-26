import os

def mostrar_estructura(ruta, limite=3):
    for raiz, carpetas, archivos in os.walk(ruta):
        # Calcular el nivel de indentación
        nivel = raiz.replace(ruta, '').count(os.sep)
        indentacion = ' ' * 2 * nivel
        print(f"{indentacion}{os.path.basename(raiz)}/")
        
        # Combinar carpetas y archivos para mostrar solo los primeros N
        contenido = carpetas + archivos
        sub_indentacion = ' ' * 2 * (nivel + 1)
        
        for item in contenido[:limite]:
            print(f"{sub_indentacion}> {item}")

# Usa '.' para la carpeta actual o pon una ruta específica
mostrar_estructura('./', limite=3)

