import pandas as pd
import re

def parse_prerequisites(prereq_text):
    """
    Parsea el texto de prerrequisitos y devuelve una lista de opciones.
    Cada opción es una lista de materias que deben cumplirse (AND).
    """
    if pd.isna(prereq_text) or prereq_text.strip() == '':
        return []
    
    # Limpiar el texto
    text = str(prereq_text).strip()
    
    # Reemplazar O y A por símbolos más fáciles de procesar
    text = text.replace(' O ', ' | ')
    text = text.replace(' A ', ' & ')
    
    # Función recursiva para procesar expresiones con paréntesis
    def evaluate_expression(expr):
        expr = expr.strip()
        
        # Si no hay paréntesis, procesar directamente
        if '(' not in expr:
            return process_simple_expression(expr)
        
        # Encontrar el paréntesis más interno
        while '(' in expr:
            # Buscar el último paréntesis de apertura
            start = expr.rfind('(')
            # Buscar el primer paréntesis de cierre después del de apertura
            end = expr.find(')', start)
            
            if end == -1:
                break
                
            # Extraer la expresión dentro de los paréntesis
            inner_expr = expr[start+1:end]
            inner_result = process_simple_expression(inner_expr)
            
            # Reemplazar la expresión entre paréntesis con un placeholder
            placeholder = f"TEMP_{len(inner_result)}_OPTIONS"
            expr = expr[:start] + placeholder + expr[end+1:]
            
            # Guardar el resultado temporalmente
            temp_results[placeholder] = inner_result
        
        return process_simple_expression(expr)
    
    def process_simple_expression(expr):
        # Dividir por OR (|)
        or_parts = [part.strip() for part in expr.split('|')]
        result = []
        
        for or_part in or_parts:
            if 'TEMP_' in or_part and '_OPTIONS' in or_part:
                # Es un placeholder, recuperar los resultados temporales
                result.extend(temp_results[or_part.strip()])
            else:
                # Dividir por AND (&)
                and_parts = [part.strip() for part in or_part.split('&')]
                # Filtrar partes vacías
                and_parts = [part for part in and_parts if part]
                if and_parts:
                    result.append(and_parts)
        
        return result
    
    temp_results = {}
    options = evaluate_expression(text)
    
    return options

def expand_and_combinations(options_list, temp_results):
    """
    Expande las combinaciones cuando hay ANDs con opciones múltiples
    """
    final_options = []
    
    for option in options_list:
        # Si algún elemento de la opción es un placeholder con múltiples opciones
        expanded_combinations = [option]
        
        for i, element in enumerate(option):
            if element in temp_results:
                new_combinations = []
                sub_options = temp_results[element]
                
                for combo in expanded_combinations:
                    for sub_option in sub_options:
                        new_combo = combo.copy()
                        new_combo[i:i+1] = sub_option  # Reemplazar el elemento
                        new_combinations.append(new_combo)
                
                expanded_combinations = new_combinations
        
        final_options.extend(expanded_combinations)
    
    return final_options

def process_prerequisites_file(file_path):
    """
    Procesa el archivo de prerrequisitos y genera las nuevas columnas
    """
    # Leer el archivo
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    
    # Verificar que las columnas necesarias existen
    if 'prereq completo' not in df.columns:
        raise ValueError("La columna 'prereq completo' no se encuentra en el archivo")
    
    # Procesar cada fila
    all_options = []
    max_options = 0
    
    for idx, row in df.iterrows():
        prereq_text = row['prereq completo']
        options = parse_prerequisites(prereq_text)
        
        # Convertir las opciones a strings con &
        formatted_options = []
        for option in options:
            if len(option) > 1:
                formatted_options.append(' & '.join(option))
            elif len(option) == 1:
                formatted_options.append(option[0])
        
        all_options.append(formatted_options)
        max_options = max(max_options, len(formatted_options))
    
    # Crear las nuevas columnas
    for i in range(max_options):
        col_name = f'Opcion_Prereq_{i+1}'
        df[col_name] = ''
        
        for idx, options in enumerate(all_options):
            if i < len(options):
                df.loc[idx, col_name] = options[i]
    
    return df

def main():
    """
    Función principal para ejecutar el procesamiento
    """
    # Solicitar la ruta del archivo
    file_path = input("Ingresa la ruta del archivo (CSV o Excel): ").strip()
    
    try:
        # Procesar el archivo
        df_result = process_prerequisites_file(file_path)
        
        # Mostrar información sobre el resultado
        print(f"\nProcesamiento completado!")
        print(f"Total de filas: {len(df_result)}")
        
        # Contar cuántas columnas de opciones se crearon
        option_cols = [col for col in df_result.columns if col.startswith('Opcion_Prereq_')]
        print(f"Columnas de opciones creadas: {len(option_cols)}")
        
        # Mostrar algunas filas de ejemplo
        print("\nPrimeras 5 filas con las nuevas columnas:")
        cols_to_show = ['Smbarul_Key_Rule', 'prereq completo'] + option_cols
        available_cols = [col for col in cols_to_show if col in df_result.columns]
        print(df_result[available_cols].head())
        
        # Guardar el resultado
        output_path = file_path.rsplit('.', 1)[0] + '_procesado.' + file_path.rsplit('.', 1)[1]
        if file_path.endswith('.csv'):
            df_result.to_csv(output_path, index=False)
        else:
            df_result.to_excel(output_path, index=False)
        
        print(f"\nArchivo guardado como: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    return df_result

# Función de prueba con ejemplos
def test_parser():
    """
    Función para probar el parser con ejemplos
    """
    test_cases = [
        "MAT1101 O INTE 00",
        "MAT1111 A ( FIS1023 O CSV0040 ) O INTE 00",
        "FIS1001 A MAT1001",
        "( QUI1001 O BIO1001 ) A MAT1002",
        ""
    ]
    
    print("Pruebas del parser:")
    for test in test_cases:
        options = parse_prerequisites(test)
        formatted_options = []
        for option in options:
            if len(option) > 1:
                formatted_options.append(' & '.join(option))
            elif len(option) == 1:
                formatted_options.append(option[0])
        
        print(f"Input: '{test}'")
        print(f"Output: {formatted_options}")
        print("-" * 50)

if __name__ == "__main__":
    # Descomentar la siguiente línea para ejecutar pruebas
    # test_parser()
    
    # Ejecutar el procesamiento principal
    main()