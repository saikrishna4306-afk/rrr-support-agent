"""20 questions to measure the real system (bge-m3). Run:  python -m tests.run_eval

expected = topic name of the guide that should be found (language suffix ignored),
or None when the question is NOT covered by the guides and must go to a human.
"""
QUESTIONS = [
    ("My upper thread keeps breaking while embroidering", "thread_breaking"),
    ("needle breaks again and again, what should I do?", "needle_breaking"),
    ("the machine is skipping stitches", "skipped_stitches"),
    ("embroidery outline is not matching the fill, there are gaps", "design_misalignment"),
    ("Error E21 is showing and the machine will not start", "error_e21"),
    ("thread loops and a nest under the fabric", "bobbin_thread_loose"),
    ("ఎంబ్రాయిడరీ చేసేటప్పుడు పై దారం తెగిపోతోంది, ఏం చేయాలి?", "thread_breaking"),
    ("సూది విరిగిపోతోంది", "needle_breaking"),
    ("कढ़ाई करते समय ऊपर का धागा बार-बार टूट रहा है", "thread_breaking"),
    ("सुई बार-बार टूट जाती है", "needle_breaking"),
    ("What is the capital of France?", None),
    ("Give me a chocolate cake recipe", None),
    ("Who won the cricket match yesterday?", None),
    ("My car engine is overheating", None),
    ("What is the price of your embroidery machine?", None),
    ("Tell me about your warranty policy", None),
    ("How do I learn Python programming?", None),
    ("My washing machine is making a loud noise", None),
    ("ఈ రోజు వాతావరణం ఎలా ఉంది?", None),
    ("आज मौसम कैसा है?", None),
]
