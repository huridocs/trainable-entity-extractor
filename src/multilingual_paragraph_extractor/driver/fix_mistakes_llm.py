from ollama import Client

text_1 = '''4.
Ante el alegado aumento e intensificación de la violencia en contra de estos pueblos,
y la falta de medidas efectivas por parte del Estado de Brasil para mitigar la situación, este
Tribunal consideró que existía un riesgo latente de que estos daños se consumaran y se
intensificaran. En virtud de ello, ordenó que (i) el Estado adoptara las medidas necesarias
para proteger efectivamente la vida, la integridad personal, la salud y el acceso a la
alimentación y al agua potable de los miembros de los Pueblos Indígenas Yanomami,
Ye’Kwana y Munduruku, desde una perspectiva culturalmente adecuada, con un enfoque de
género y edad; (ii) el Estado adoptara las medidas necesarias para prevenir la explotación y
la violencia sexual contra las mujeres y niñas de los Pueblos Indígenas beneficiarios; (iii) el
Estado adoptara las medidas culturalmente apropiadas para prevenir la propagación y mitigar
el contagio de enfermedades, especialmente del COVID-19, prestándoles a las personas
 beneficiarias una atención médica adecuada, de acuerdo con las normas internacionales
aplicables, y que (iv) el Estado adoptara las medidas necesarias para proteger la vida e
integridad personal de los líderes y lideresas indígenas de los Pueblos Indígenas beneficiarios
que se encuentran bajo amenaza.'''

text_2 = '''4.
Diante do suposto aumento e intensificação da violência contra esses povos,
e da falta de medidas efetivas por parte do Estado do Brasil para mitigar a situação,
este Tribunal considerou que havia um risco latente de que esses danos se
consumassem e se intensificassem. Em virtude disso, ordenou que (i) o Estado
adotasse as medidas necessárias para proteger efetivamente a vida, a integridade
pessoal, a saúde e o acesso à alimentação e à água potável dos membros dos Povos
Indígenas Yanomami, Ye’Kwana e Munduruku, a partir de uma perspectiva
culturalmente adequada, com abordagem de gênero e idade; (ii) o Estado adotasse
as medidas necessárias para prevenir a exploração e a violência sexual contra as
mulheres e meninas dos povos indígenas beneficiários; (iii) o Estado adotasse as
medidas culturalmente adequadas para prevenir a propagação e mitigar o contágio
de doenças, especialmente a covid-19, prestando às pessoas beneficiárias uma
atenção médica adequada, de acordo com as normas internacionais aplicáveis; e que'''

QUESTIONS = [   {
        "text": f"""You are an expert in multilingual text analysis. You will receive two paragraphs written in different languages.

Your task is to determine if these two paragraphs convey the *same core information* or not. Focus on the meaning and intent of the paragraphs, not on their literal wording or structure.

### Instructions:
1. If the two paragraphs convey the same core information, regardless of differences in phrasing or language, reply "True".
2. If the two paragraphs convey different information, reply "False".
3. Do not provide any additional explanation or commentary—only reply with "True" or "False".

### PARAGRAPH 1:
{text_1}

### PARAGRAPH 2:
{text_2}

### ANSWER (True/False):
""",
        "expected_result": "True"
    }
]

def run():
    client = Client()
    for i, question in enumerate(QUESTIONS):
        response = client.chat(model="aya-expanse:32b", messages=[{"role": "user", "content": question["text"]}])
        response_content = response["message"]["content"]
        print(f"question {i}:")
        print(question["text"][:100].replace("\n", " "))
        print("answer:")
        print(response_content)
        print("expected:")
        print(question["expected_result"])
        print()
        print()

if __name__ == '__main__':
    run()
