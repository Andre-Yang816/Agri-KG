import openai
import itertools
from openai import OpenAI
# Set OpenAI API Key

client = OpenAI(api_key="sk-20c869e157c34491a7e7eacd011bad4b", base_url="https://api.deepseek.com/v1")

# Read entities file
with open("filtered_entities.txt", "r", encoding="utf-8") as f:
    entities = [line.strip() for line in f.readlines()]

# Generate all possible entity pairs
entity_pairs = list(itertools.combinations(entities, 2))

# Known relationships (to filter out)
# known_triples = set()
# with open("baike_triples.txt", "r", encoding="utf-8") as f:
#     for line in f:
#         known_triples.add(line.strip())

# Relationship dictionary (for filtering GPT's generated relations)
# relation_dict = [
#     "影响", "促进", "抑制", "防治", "传播", "需要", "依赖", "寄生于",
#     "适应", "导致", "属于", "用于", "危害", "增强", "减少", "调控"
# ]


def extract_relations(entity_pairs):
    """Batch process multiple entity pairs to predict relationships"""
    prompt = f"""
    你是农业知识专家，正在构建农业知识图谱。请分析以下农业实体对的可能关系，并以标准三元组格式返回：

    你需要：
    1. **分析实体间的所有可能关系**（基于农业、生物学、环境和生产领域）。
    2. **识别可能的隐含实体**（如果合适，可补充缺失的关键实体）。
    3. 以 **严格标准格式** 返回： `<实体1, 关系, 实体2>`，**每行一个关系**。
    4. **不要额外的文字说明或解释**。

    **示例输出**：
    ```
    <水稻, 易感染, 稻瘟病>
    <稻瘟病, 影响, 水稻>
    <水稻, 需要, 氮肥>
    <氮肥, 促进, 水稻生长>
    ```

    现在请分析以下实体对：
    {entity_pairs}
    """

    # Use openai.completions.create() for generating the response
    # response = openai.completions.create(
    #     model="gpt-4",  # You can use "gpt-3.5-turbo" here as well
    #     prompt=prompt,
    #     max_tokens=1024,  # Limit tokens to prevent response truncation
    #     temperature=0.7  # Adjust temperature for varied outputs
    # )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": '你是一个农业知识专家，帮助构建农业知识图谱。'},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    result = response['choices'][0]['text'].strip()  # Extract the text part of the response

    # Parse the returned triples
    triples = [line.strip() for line in result.split("\n") if line.startswith("<") and line.endswith(">")]

    # Filter out known relationships and invalid ones
    # filtered_triples = [t for t in triples if t not in known_triples]
    #
    # return filtered_triples if filtered_triples else None
    return triples

# Process all entity pairs in batches
all_triples = []
batch_size = 5  # Set batch size for processing multiple pairs in one go
for i in range(0, len(entity_pairs), batch_size):
    batch = entity_pairs[i:i + batch_size]
    new_triples = extract_relations(batch)
    if new_triples:
        all_triples.extend(new_triples)

# Save the results to a file
with open("triples.txt", "w", encoding="utf-8") as f:
    for triple in all_triples:
        f.write(triple + "\n")

print(f"Generated {len(all_triples)} new agricultural relationship triples, saved to triples.txt")
