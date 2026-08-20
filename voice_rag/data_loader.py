"""
Dataset loading with resilient fallback paths for local and serverless Vercel execution.

Primary path (`load_msmarco_xi`): uses the `datasets` library to pull ai4bharat/MSMARCO-XI from HuggingFace.
Local/Offline fallback: loads from `data/sample_msmarco_xi.jsonl`.
Embedded backup: built-in list ensuring the corpus always initializes even in serverless Lambda bundle edge cases.
"""
from __future__ import annotations
import json
import os
from typing import List, Dict

# Candidate paths for dataset (prioritizing the full 10,000 row MSMARCO-XI index)
_CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(__file__), "data", "msmarco_xi_10000.jsonl"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voice_rag", "data", "msmarco_xi_10000.jsonl"),
    os.path.join(os.getcwd(), "voice_rag", "data", "msmarco_xi_10000.jsonl"),
    os.path.join(os.getcwd(), "data", "msmarco_xi_10000.jsonl"),
    os.path.join(os.path.dirname(__file__), "data", "sample_msmarco_xi.jsonl"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voice_rag", "data", "sample_msmarco_xi.jsonl"),
    os.path.join(os.getcwd(), "voice_rag", "data", "sample_msmarco_xi.jsonl"),
    os.path.join(os.getcwd(), "data", "sample_msmarco_xi.jsonl"),
]

SAMPLE_PATH = _CANDIDATE_PATHS[0]
for p in _CANDIDATE_PATHS:
    if os.path.exists(p):
        SAMPLE_PATH = p
        break


# Embedded sample dataset fallback in case filesystem packaging isolates .jsonl files on Vercel
_EMBEDDED_DOCS = [
    {"id": "d001", "language": "hi", "query": "मधुमेह के लक्षण क्या हैं?", "passage": "मधुमेह (डायबिटीज) एक ऐसी स्थिति है जिसमें रक्त में शर्करा (ग्लूकोज) का स्तर सामान्य से अधिक हो जाता है। इसके सामान्य लक्षणों में अत्यधिक प्यास लगना, बार-बार पेशाब आना, अत्यधिक भूख लगना, वजन कम होना, थकान और धुंधला दिखाई देना शामिल हैं। टाइप 1 और टाइप 2 मधुमेह में लक्षण अलग-अलग गति से विकसित होते हैं। समय पर निदान और उपचार जटिलताओं को रोकने में मदद करता है।"},
    {"id": "d002", "language": "en", "query": "what are the symptoms of diabetes", "passage": "Diabetes is a chronic condition that affects how the body turns food into energy. Common symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, and blurred vision. Type 1 diabetes symptoms can develop quickly, while type 2 diabetes symptoms often develop slowly over several years. Early diagnosis through blood glucose testing helps prevent long-term complications such as nerve damage and kidney disease."},
    {"id": "d003", "language": "hi", "query": "भारत की राजधानी कौन सी है?", "passage": "नई दिल्ली भारत की राजधानी है। यह राष्ट्रीय राजधानी क्षेत्र (एनसीआर) का हिस्सा है और भारत सरकार की तीनों शाखाओं - कार्यपालिका, विधायिका और न्यायपालिका - का मुख्यालय यहीं स्थित है। दिल्ली को 1911 में ब्रिटिश भारत की राजधानी बनाया गया था और आज यह देश का प्रमुख राजनीतिक, सांस्कृतिक और वाणिज्यिक केंद्र है।"},
    {"id": "d004", "language": "en", "query": "what is the capital of india", "passage": "New Delhi is the capital of India and forms part of the National Capital Territory. It houses the executive, legislative, and judicial branches of the Government of India. Delhi became the capital of British India in 1911, and today it stands as one of the country's major political, cultural, and commercial hubs, alongside Mumbai and Bengaluru."},
    {"id": "d005", "language": "hi", "query": "रिजर्व बैंक ऑफ इंडिया की स्थापना कब हुई?", "passage": "भारतीय रिजर्व बैंक (आरबीआई) की स्थापना 1 अप्रैल 1935 को भारतीय रिजर्व बैंक अधिनियम, 1934 के तहत की गई थी। यह भारत का केंद्रीय बैंक है और मुद्रा जारी करने, मौद्रिक नीति बनाने और बैंकिंग प्रणाली को विनियमित करने के लिए जिम्मेदार है। इसका मुख्यालय मुंबई में स्थित है।"},
    {"id": "d006", "language": "en", "query": "when was the reserve bank of india established", "passage": "The Reserve Bank of India (RBI) was established on 1 April 1935 under the Reserve Bank of India Act, 1934. It serves as the central bank of India, responsible for issuing currency, formulating monetary policy, and regulating the banking system. Its headquarters is located in Mumbai."},
    {"id": "d007", "language": "hi", "query": "प्रकाश संश्लेषण की प्रक्रिया क्या है?", "passage": "प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा पौधे, शैवाल और कुछ जीवाणु सूर्य के प्रकाश का उपयोग करके कार्बन डाइऑक्साइड और पानी से ग्लूकोज और ऑक्सीजन का निर्माण करते हैं। यह प्रक्रिया पौधों की कोशिकाओं में क्लोरोप्लास्ट के अंदर होती है और पृथ्वी पर अधिकांश जीवन के लिए ऊर्जा का प्राथमिक स्रोत है।"},
    {"id": "d008", "language": "en", "query": "explain the process of photosynthesis", "passage": "Photosynthesis is the process by which plants, algae, and certain bacteria convert carbon dioxide and water into glucose and oxygen using energy from sunlight. This process occurs inside chloroplasts within plant cells and is the primary energy source for most life on Earth. It consists of two main stages: the light-dependent reactions and the Calvin cycle."},
    {"id": "d009", "language": "hi", "query": "ताजमहल का निर्माण किसने करवाया था?", "passage": "ताजमहल का निर्माण मुगल सम्राट शाहजहाँ ने अपनी पत्नी मुमताज महल की याद में करवाया था। इसका निर्माण कार्य 1632 में शुरू हुआ और लगभग 1653 में पूरा हुआ। यह आगरा, उत्तर प्रदेश में यमुना नदी के किनारे स्थित है और इसे यूनेस्को विश्व धरोहर स्थल घोषित किया गया है।"},
    {"id": "d010", "language": "en", "query": "who built the taj mahal", "passage": "The Taj Mahal was built by the Mughal emperor Shah Jahan in memory of his wife Mumtaz Mahal. Construction began around 1632 and was largely completed by 1653. Located in Agra, Uttar Pradesh, on the banks of the Yamuna River, it is recognized as a UNESCO World Heritage Site and a symbol of Mughal architecture."},
    {"id": "d011", "language": "hi", "query": "कंप्यूटर की मुख्य मेमोरी क्या है?", "passage": "रैंडम एक्सेस मेमोरी (रैम) कंप्यूटर की मुख्य मेमोरी है जिसका उपयोग वर्तमान में चल रहे प्रोग्रामों और डेटा को अस्थायी रूप से संग्रहीत करने के लिए किया जाता है। यह वाष्पशील मेमोरी है, यानी बिजली बंद होने पर इसमें संग्रहीत डेटा नष्ट हो जाता है। रैम की गति हार्ड डिस्क की तुलना में बहुत अधिक होती है।"},
    {"id": "d012", "language": "en", "query": "what is the main memory of a computer", "passage": "Random Access Memory (RAM) is the main memory of a computer, used to temporarily store data and programs that are currently in use. RAM is volatile memory, meaning its contents are lost when power is turned off. It is significantly faster than secondary storage such as hard disks or SSDs, which enables quick read and write access for the CPU."},
    {"id": "d013", "language": "hi", "query": "योग के क्या लाभ हैं?", "passage": "योग करने से शारीरिक लचीलापन बढ़ता है, मांसपेशियां मजबूत होती हैं और तनाव कम होता है। नियमित योगाभ्यास रक्तचाप नियंत्रित करने, नींद की गुणवत्ता सुधारने और मानसिक स्वास्थ्य को बेहतर बनाने में मदद करता है। यह हृदय स्वास्थ्य और समग्र कल्याण के लिए भी लाभकारी माना जाता है।"},
    {"id": "d014", "language": "en", "query": "what are the benefits of yoga", "passage": "Practicing yoga improves physical flexibility, builds muscle strength, and reduces stress levels. Regular yoga practice helps regulate blood pressure, improves sleep quality, and enhances mental well-being. It is also considered beneficial for cardiovascular health and overall wellness, and is often combined with breathing exercises and meditation."},
    {"id": "d015", "language": "hi", "query": "जलवायु परिवर्तन के मुख्य कारण क्या हैं?", "passage": "जलवायु परिवर्तन का मुख्य कारण ग्रीनहाउस गैसों का उत्सर्जन है, जो जीवाश्म ईंधन जलाने, वनों की कटाई और औद्योगिक गतिविधियों से उत्पन्न होती हैं। कार्बन डाइऑक्साइड और मीथेन जैसी गैसें वायुमंडल में गर्मी को रोकती हैं, जिससे वैश्विक तापमान बढ़ता है और मौसम के पैटर्न में बदलाव आता है।"},
    {"id": "d016", "language": "en", "query": "what are the main causes of climate change", "passage": "The main cause of climate change is the emission of greenhouse gases, primarily from burning fossil fuels, deforestation, and industrial activities. Gases such as carbon dioxide and methane trap heat in the atmosphere, causing global temperatures to rise and altering weather patterns, sea levels, and ecosystems worldwide."},
    {"id": "d017", "language": "hi", "query": "भारतीय संविधान कब लागू हुआ?", "passage": "भारत का संविधान 26 जनवरी 1950 को लागू हुआ था, इसीलिए इस दिन को गणतंत्र दिवस के रूप में मनाया जाता है। संविधान सभा ने 26 नवंबर 1949 को इसे अंगीकार किया था। डॉ. भीमराव अंबेडकर को भारतीय संविधान का जनक माना जाता है, क्योंकि उन्होंने प्रारूप समिति के अध्यक्ष के रूप में महत्वपूर्ण भूमिका निभाई थी।"},
    {"id": "d018", "language": "en", "query": "when did the indian constitution come into effect", "passage": "The Constitution of India came into effect on 26 January 1950, which is why the day is celebrated as Republic Day. It was adopted by the Constituent Assembly on 26 November 1949. Dr. B. R. Ambedkar is regarded as the chief architect of the Indian Constitution for his role as chairman of the Drafting Committee."},
    {"id": "d019", "language": "hi", "query": "सौर मंडल में कितने ग्रह हैं?", "passage": "सौर मंडल में आठ ग्रह हैं: बुध, शुक्र, पृथ्वी, मंगल, बृहस्पति, शनि, अरुण और वरुण। 2006 में अंतर्राष्ट्रीय खगोलीय संघ ने प्लूटो को बौने ग्रह की श्रेणी में रखा, जिसके बाद ग्रहों की संख्या नौ से घटकर आठ रह गई। बृहस्पति सौर मंडल का सबसे बड़ा ग्रह है।"},
    {"id": "d020", "language": "en", "query": "how many planets are in the solar system", "passage": "There are eight planets in the solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. In 2006, the International Astronomical Union reclassified Pluto as a dwarf planet, reducing the count from nine to eight. Jupiter is the largest planet, while Mercury is the smallest and closest to the Sun."},
    {"id": "d021", "language": "hi", "query": "ब्लॉकचेन तकनीक कैसे काम करती है?", "passage": "ब्लॉकचेन एक विकेंद्रीकृत डिजिटल बहीखाता प्रणाली है जो लेनदेन को कई कंप्यूटरों के नेटवर्क में रिकॉर्ड करती है। प्रत्येक ब्लॉक में लेनदेन का डेटा, एक टाइमस्टैम्प और पिछले ब्लॉक का क्रिप्टोग्राफिक हैश होता है, जिससे ब्लॉकों की एक श्रृंखला बनती है। इसे बदलना अत्यंत कठिन होता है, जो इसे सुरक्षित और पारदर्शी बनाता है।"},
    {"id": "d022", "language": "en", "query": "how does blockchain technology work", "passage": "Blockchain is a decentralized digital ledger system that records transactions across a network of computers. Each block contains transaction data, a timestamp, and a cryptographic hash of the previous block, forming a chain. This structure makes the ledger extremely difficult to alter, providing security and transparency without a central authority."},
    {"id": "d023", "language": "hi", "query": "गंगा नदी की लंबाई कितनी है?", "passage": "गंगा नदी की कुल लंबाई लगभग 2,525 किलोमीटर है। यह उत्तराखंड के गंगोत्री ग्लेशियर से निकलती है और उत्तर प्रदेश, बिहार, झारखंड और पश्चिम बंगाल से होते हुए बंगाल की खाड़ी में जाकर मिलती है। यह भारत की सबसे पवित्र और सांस्कृतिक रूप से महत्वपूर्ण नदी मानी जाती है।"},
    {"id": "d024", "language": "en", "query": "what is the length of the ganges river", "passage": "The Ganges River is approximately 2,525 kilometers long. It originates from the Gangotri Glacier in Uttarakhand and flows through Uttar Pradesh, Bihar, Jharkhand, and West Bengal before emptying into the Bay of Bengal. It is considered the most sacred and culturally significant river in India."},
    {"id": "d025", "language": "hi", "query": "उच्च रक्तचाप को कैसे नियंत्रित करें?", "passage": "उच्च रक्तचाप को नियंत्रित करने के लिए नमक का सेवन कम करना, नियमित व्यायाम करना, स्वस्थ वजन बनाए रखना, शराब और धूम्रपान से बचना और तनाव प्रबंधन करना महत्वपूर्ण है। कुछ मामलों में डॉक्टर दवाइयां भी लिख सकते हैं। नियमित रूप से रक्तचाप की जांच कराना आवश्यक है।"},
    {"id": "d026", "language": "en", "query": "how to control high blood pressure", "passage": "Controlling high blood pressure involves reducing salt intake, exercising regularly, maintaining a healthy weight, limiting alcohol consumption, avoiding smoking, and managing stress. In some cases, doctors may prescribe medication such as ACE inhibitors or diuretics. Regular blood pressure monitoring is essential for managing hypertension effectively."},
    {"id": "d027", "language": "hi", "query": "भारत में स्वतंत्रता दिवस कब मनाया जाता है?", "passage": "भारत में स्वतंत्रता दिवस हर वर्ष 15 अगस्त को मनाया जाता है। 15 अगस्त 1947 को भारत को ब्रिटिश शासन से स्वतंत्रता मिली थी। इस दिन देश भर में झंडा फहराने के कार्यक्रम, सांस्कृतिक कार्यक्रम आयोजित किए जाते हैं और प्रधानमंत्री लाल किले से राष्ट्र को संबोधित करते हैं।"},
    {"id": "d028", "language": "en", "query": "when is independence day celebrated in india", "passage": "Independence Day is celebrated in India every year on 15 August. India gained independence from British rule on 15 August 1947. The day is marked with flag hoisting ceremonies, cultural programs across the country, and an address to the nation by the Prime Minister from the Red Fort in Delhi."},
    {"id": "d029", "language": "hi", "query": "5जी तकनीक की मुख्य विशेषताएं क्या हैं?", "passage": "5जी तकनीक की मुख्य विशेषताओं में अत्यधिक तेज डेटा गति, कम विलंबता (लेटेंसी), और एक साथ अधिक उपकरणों को जोड़ने की क्षमता शामिल है। यह 4जी की तुलना में दस गुना तेज हो सकती है और स्वचालित वाहनों, स्मार्ट शहरों और रिमोट सर्जरी जैसे अनुप्रयोगों को सक्षम बनाती है।"},
    {"id": "d030", "language": "en", "query": "what are the key features of 5g technology", "passage": "The key features of 5G technology include significantly faster data speeds, ultra-low latency, and the ability to connect a much larger number of devices simultaneously compared to 4G. 5G can be up to ten times faster than 4G and enables applications such as autonomous vehicles, smart cities, and remote surgery."}
]


def load_msmarco_xi(split: str = "train", language: str = "hi", limit: int | None = 500):
    """Real loader — requires network access to huggingface.co."""
    from datasets import load_dataset  # imported lazily; optional dependency
    ds = load_dataset("ai4bharat/MSMARCO-XI", language, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def load_docs(limit: int | None = None) -> List[Dict]:
    """Returns a list of {id, text, language, metadata} dicts ready for chunking.
    Tries HF dataset first, falls back to sample file, then embedded backup."""
    try:
        ds = load_msmarco_xi(limit=limit or 500)
        docs = []
        for i, row in enumerate(ds):
            docs.append({
                "id": f"hf_{i}",
                "text": row.get("passage") or row.get("text") or "",
                "language": row.get("language", "hi"),
                "metadata": {"query": row.get("query", ""), "source": "ai4bharat/MSMARCO-XI"},
            })
        return docs
    except Exception:
        return load_sample_docs(limit=limit)


def load_sample_docs(limit: int | None = None) -> List[Dict]:
    docs = []
    # Try reading from file first
    if os.path.exists(SAMPLE_PATH):
        try:
            with open(SAMPLE_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        docs.append({
                            "id": row["id"],
                            "text": row["passage"],
                            "language": row["language"],
                            "metadata": {"query": row["query"], "source": "sample_msmarco_xi"},
                        })
            if docs:
                return docs[:limit] if limit else docs
        except Exception:
            pass

    # Embedded fallback guarantee
    for row in _EMBEDDED_DOCS:
        docs.append({
            "id": row["id"],
            "text": row["passage"],
            "language": row["language"],
            "metadata": {"query": row["query"], "source": "sample_msmarco_xi"},
        })
    return docs[:limit] if limit else docs


def load_sample_queries(limit: int | None = None) -> List[Dict]:
    """Sample queries paired with expected document IDs."""
    sample_data = _EMBEDDED_DOCS
    if os.path.exists(SAMPLE_PATH):
        try:
            file_data = []
            with open(SAMPLE_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        file_data.append(json.loads(line))
            if file_data:
                sample_data = file_data
        except Exception:
            pass

    queries = []
    for row in sample_data:
        queries.append({"query": row["query"], "expected_doc_id": row["id"], "language": row["language"]})
    return queries[:limit] if limit else queries
