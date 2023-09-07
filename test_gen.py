from prompt2model.dataset_generator import OpenAIDatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType

api_key = "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"
dataset_generator = OpenAIDatasetGenerator(api_key)

prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
# prompt = """Your task is to translate English sentences to Javanese. In this task, you are given a sentence in English which you are to translate to Javanese.

# Here are examples with English input sentences, along with their expected Javanese text outputs:

# input="I like to go swimming with my friends every weekend"
# output="Aku seneng budal renang karo kanca-kancaku saben akhir minggu"

# input="I like to eat fried rice, chicken satay, and soto bandung"
# output="Aku seneng mangan sego goreng, sate ayam, karo soto bandung"

# input="I like to relax at the hotel room and enjoy the view we have there"
# output="Aku seneng bersantai nang kamar hotel karo nikmati pemandangan sing nang kono"
# """

prompt = """Your task is to translate English sentences to Javanese. In this task, you are given a sentence in English which you are to translate to Javanese.

Here are examples with English input sentences, along with their expected Javanese text outputs:

input="On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."
output="Ing dina Senin, ilmuwan saka Fakultas Kedokteran Universitas Stanford ngumumake penemuan piranti diagnosa sing isa misah-misahake sel adhedhasar jenis: kepingan cilik sing isa dicetak sing isa diasilake kanthi nggunakake printer inkjet standar kanggo mungkin saben-saben watara saji sen AS."

input="Lead researchers say this may bring early detection of cancer, tuberculosis, HIV and malaria to patients in low-income countries, where the survival rates for illnesses such as breast cancer can be half those of richer countries."
output="Panliti kang mimpin ngomong menawa iki bisa nyebabake deteksi awal kanggo kanker, tuberkulosis, HIV lan malaria kanggo pasien ing negara pendapatan sithik, ing ngendi tingkat kaslametan kanggo penyakit kaya kanker payudara bisa separone saka negara kang luwih sugih."

input="The JAS 39C Gripen crashed onto a runway at around 9:30 am local time (0230 UTC) and exploded, closing the airport to commercial flights."
output="JAS 39C Gripen tiba ing dalan kira-kira jam 9.30 esuk wektu lokal (0230 UTC) lan njeblug, nutup bendara kanggo panerbangan komersil."
"""
prompt_spec.parse_from_prompt(prompt)

expected_num_examples = 10
split = DatasetSplit.TRAIN
dataset = dataset_generator.generate_dataset_split(
    prompt_spec
    , expected_num_examples
    , split
    )
dataset.save_to_disk("generated_dataset_highq_examples")