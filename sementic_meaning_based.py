from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    embeddings=GoogleGenerativeAIEmbeddings('gemini-embedding-exp-03-07'),
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold=1,
)

text = '''
Artificial intelligence is rapidly transforming industries—from healthcare diagnostics to automated legal analysis. In parallel, social media continues to shape public opinion at an unprecedented scale, blurring the lines between truth and misinformation. While AI models become more capable, ethical concerns about bias, transparency, and data privacy grow louder.

Meanwhile, space exploration has entered a new era. Private companies are launching satellites, planning lunar missions, and even targeting Mars as a future human destination. This renewed interest in space is not just scientific; it’s also geopolitical and commercial.

At the same time, education systems worldwide are grappling with the shift to hybrid learning. Many institutions have embraced digital platforms, but questions remain about accessibility, attention spans, and long-term learning outcomes. The digital divide remains a serious barrier for millions of students.

As these threads converge, policymakers must consider how technology, science, and society intersect. Future progress depends not just on innovation, but also on inclusive, ethical, and forward-thinking governance.
'''

result = text_splitter.create_documents([text])

print(len(result))
