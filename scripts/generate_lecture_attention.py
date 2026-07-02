from docx import Document
from docx.shared import Pt, Cm, RGBColor
import os

doc = Document()

style = doc.styles["Normal"]
font = style.font
font.Name = "Times New Roman"
font.Size = Pt(12)

section = doc.sections[0]
section.left_margin = Cm(2.5)
section.right_margin = Cm(2.5)
section.top_margin = Cm(2)
section.bottom_margin = Cm(2)

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0, 0, 0)
        sz = 16 if level == 1 else 14 if level == 2 else 12
        run.font.size = Pt(sz)
    return h

def add_para(text, bold=False, italic=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.first_line_indent = Cm(1.25)
    return p

def add_code_block(code):
    p = doc.add_paragraph()
    run = p.add_run(code)
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(50, 50, 50)
    p.paragraph_format.left_indent = Cm(1)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    return p

# =============== TITLE ===============
title = doc.add_heading("Практическая реализация механизма внимания", level=0)
for run in title.runs:
    run.font.color.rgb = RGBColor(0, 0, 0)

add_para("Документация кода: Self-Attention, Multi-Head Attention и GQA для проекта FishingLLM")
add_para("")

# =============== 1 ===============
add_heading("1. Общая архитектура модуля")

add_para(
    "Модуль fishingllm/attention.py содержит два класса: ScaledDotProductAttention и "
    "MultiHeadAttention. Первый реализует базовую формулу скалярного произведения с "
    "масштабированием, второй — многоголовое внимание с поддержкой Grouped Query "
    "Attention (GQA). Оба наследуются от nn.Module, что позволяет встраивать их "
    "в стандартный PyTorch-пайплайн."
)

add_para(
    "Напомним формулу attention из Лекции 1: "
    "Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V. "
    "Q — запросы (queries), K — ключи (keys), V — значения (values), "
    "d_k — размерность ключей, sqrt(d_k) — нормировочный коэффициент, "
    "предотвращающий взрыв softmax при больших размерностях."
)

# =============== 2 ===============
add_heading("2. ScaledDotProductAttention — одно головое внимание", level=2)

add_para(
    "Класс состоит из конструктора и метода forward. В конструкторе запоминаем d_k "
    "и предвычисляем scale = 1/sqrt(d_k). Предвычисление — микрооптимизация: "
    "вместо 1/math.sqrt(d_k) на каждом шаге forward считаем один раз в __init__."
)

add_para("Рассмотрим forward-метод построчно:", bold=True)

add_code_block("def forward(self, Q, K, V, mask=None):\n"
    "    scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale")

add_para(
    "torch.matmul(Q, K.transpose(-2, -1)) вычисляет Q*K^T. transpose(-2, -1) "
    "транспонирует две последние оси K: если K формы (batch, seq_len, d_k), "
    "то после транспонирования получаем (batch, d_k, seq_len), и matmul даёт "
    "(batch, seq_len, seq_len). Умножение на self.scale — деление на sqrt(d_k)."
)

add_code_block("if mask is not None:\n    scores = scores.masked_fill(mask, float('-inf'))")

add_para(
    "Маска — тензор bool той же формы (batch, seq_len, seq_len). Где mask=True, "
    "ставится -inf. После softmax эти позиции дадут 0, то есть замаскированные "
    "токены не участвуют. Для каузальной маски запрещаем attendить к будущим "
    "токенам. Почему -inf, а не большое число? Softmax с -inf даёт ровный 0, "
    "а с конечным числом — ненулевую вероятность."
)

add_code_block("attn = F.softmax(scores, dim=-1)")

add_para(
    "Softmax по последней оси (dim=-1) нормализует веса для каждого токена, "
    "превращая scores в распределение вероятностей."
)

add_code_block("return torch.matmul(attn, V), attn")

add_para(
    "Умножаем attn на V, получая взвешенную сумму значений. "
    "Возвращаем результат (batch, seq_len, d_k) и веса attn "
    "(batch, seq_len, seq_len) для отладки."
)

# =============== 3 ===============
add_heading("3. MultiHeadAttention — многоголовое внимание", level=2)

add_para(
    "MultiHeadAttention расширяет ScaledDotProductAttention до нескольких голов. "
    "Вместо одного attention параллельно вычисляем несколько, каждая на своей "
    "проекции Q, K, V."
)

add_heading("3.1 Конструктор и параметры", level=3)

add_para(
    "Параметры: d_model — размерность эмбеддингов (в проекте 768); "
    "n_heads — количество голов (12); n_kv_heads — KV-голов для GQA (4, "
    "по умолчанию равно n_heads); dropout — вероятность обнуления."
)

add_para(
    "Проверка assert d_model % n_heads == 0: если d_model не делится на n_heads, "
    "нельзя равномерно распределить размерность по головам."
)

add_para("Четыре линейные проекции:", bold=True)
add_para(
    "- W_q: d_model -> n_heads * d_k (запросы Q, полные n_heads)\n"
    "- W_k: d_model -> n_kv_heads * d_k (ключи K, только n_kv_heads)\n"
    "- W_v: d_model -> n_kv_heads * d_k (значения V, только n_kv_heads)\n"
    "- W_o: n_heads * d_k -> d_model (обратная проекция)"
)

add_para(
    "bias=False — в оригинальной статье и LLaMA bias в проекциях не используется. "
    "Уменьшает число параметров, сдвиги вносятся нормализацией слоя."
)

add_para(
    "n_rep = n_heads // n_kv_heads — коэффициент повторения KV-голов. "
    "При 12:4, n_rep=3: каждая KV-голова обслуживает 3 Q-головы. "
    "Это суть GQA: вычисляем K и V только для 4 голов, повторяем 3 раза "
    "для 12 Q-голов."
)

add_heading("3.2 Инициализация весов", level=3)

add_para(
    "Метод _reset_parameters использует Xavier Uniform инициализацию. "
    "Выбор Xavier обоснован: он поддерживает дисперсию сигнала на постоянном "
    "уровне при проходе через линейный слой."
)

add_heading("3.3 Каузальная маска", level=3)

add_code_block("@staticmethod\n"
    "def _create_causal_mask(seq_len, device):\n"
    "    mask = torch.triu(torch.full((seq_len, seq_len), True,\n"
    "        dtype=torch.bool, device=device), diagonal=1)\n"
    "    return mask.unsqueeze(0)")

add_para(
    "torch.triu с diagonal=1 берёт строго верхний треугольник (элементы выше "
    "главной диагонали) в True. Для матрицы 4x4: первые элемент (0,1), (0,2), "
    "(0,3), (1,2), (1,3), (2,3) равны True. Это означает: i-й токен attendит "
    "только к j <= i. unsqueeze(0) добавляет batch-размерность."
)

add_heading("3.4 Повтор KV-голов для GQA", level=3)

add_para(
    "В forward-методе K и V повторяются с помощью repeat_interleave:"
)

add_code_block("K = K.repeat_interleave(self.n_rep, dim=1)\nV = V.repeat_interleave(self.n_rep, dim=1)")

add_para(
    "repeat_interleave по dim=1 (размерность голов) копирует каждую KV-голову "
    "n_rep раз. Форма меняется с (B, n_kv_heads, L, d_k) на "
    "(B, n_kv_heads * n_rep, L, d_k) = (B, n_heads, L, d_k)."
)

add_heading("3.5 Forward-метод", level=3)

add_para("Шаг 1. Проекции Q, K, V:", bold=True)

add_code_block("Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)\n"
    "K = self.W_k(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)\n"
    "V = self.W_v(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)")

add_para(
    "После transpose(1, 2) форма Q — (B, n_heads, L, d_k), "
    "K и V — (B, n_kv_heads, L, d_k). Переставляем оси, чтобы "
    "размерность последовательности (L) была предпоследней."
)

add_para("Шаг 2. Применение attention ко всем головам:", bold=True)

add_code_block("attn_output, attn_weights = self.attn(\n"
    "    Q.reshape(-1, L, self.d_k),\n"
    "    K.reshape(-1, L, self.d_k),\n"
    "    V.reshape(-1, L, self.d_k), mask)")

add_para(
    "reshape(-1, L, d_k) объединяет batch и n_heads: из (B, n_heads, L, d_k) "
    "получаем (B * n_heads, L, d_k). Это превращает multi-head в batch-обработку "
    "независимых single-head attention."
)

add_para("Шаг 3. Обратная проекция:", bold=True)

add_code_block(
    "attn_output = attn_output.view(B, self.n_heads, L, self.d_k)\n"
    "    .transpose(1, 2).reshape(B, L, -1)\n"
    "attn_output = self.dropout(self.W_o(attn_output))")

add_para(
    "view восстанавливает размерность голов, transpose возвращает оси "
    "в (B, L, n_heads, d_k), reshape собирает головы в последнюю ось. "
    "W_o отображает n_heads * d_k обратно в d_model."
)

# =============== 4 ===============
add_heading("4. GQA — зачем и почему", level=2)

add_para(
    "Grouped Query Attention (Ainslie et al., 2023) — компромисс между "
    "Multi-Query Attention (MQA) и Multi-Head Attention (MHA)."
)

add_para("Сравнение:", bold=True)

add_para(
    "- MQA: n_kv_heads = 1. Максимальная экономия KV-кэша, "
    "но снижение качества (Shazeer, 2019).\n"
    "- MHA: n_kv_heads = n_heads. Максимальная ёмкость, "
    "но дорого на inference (кэш всех голов).\n"
    "- GQA: промежуточный вариант. Для 12:4 — трёхкратное сжатие "
    "KV-кэша при минимальной потере качества (< 1% perplexity)."
)

add_para(
    "Для FishingLLM на CPU с 8GB RAM каждый сэкономленный мегабайт KV-кэша "
    "позволяет обрабатывать более длинные промпты. При n_kv_heads=4 экономим "
    "2/3 памяти кэша по сравнению с полным MHA."
)

# =============== 5 ===============
add_heading("5. Обоснование тестов", level=2)

add_para(
    "Тесты разделены на два класса: TestScaledDotProductAttention (5 тестов) "
    "и TestMultiHeadAttention (7 тестов)."
)

add_heading("5.1 ScaledDotProductAttention", level=3)

add_para(
    "test_output_shape — проверяет, что выход имеет форму (B, L, d_k) и "
    "веса — (B, L, L). Минимальная проверка тензорных операций."
)
add_para(
    "test_attention_weights_sum_to_one — веса attention для каждого токена "
    "суммируются в 1 (свойство softmax)."
)
add_para(
    "test_causal_mask_prevents_future_attention — weights[i, j] = 0 для j > i. "
    "Гарантирует, что модель не подглядывает в будущие токены."
)
add_para(
    "test_identical_qkv_gives_diagonal_attention — при Q=K=V скалярное "
    "произведение токена с собой даёт максимум, диагональ доминирует (> 0.5)."
)
add_para(
    "test_scaling_factor — масштабирование sqrt(d_k) предотвращает NaN "
    "при больших значениях."
)

add_heading("5.2 MultiHeadAttention", level=3)

add_para(
    "test_output_shape — на входе (B, L, d_model), на выходе то же."
)
add_para(
    "test_gqa_kv_repeat — проверяет n_rep=2 для конфигурации 8:4, "
    "и что attention-веса имеют размерность (B, n_heads, L, L)."
)
add_para(
    "test_causal_mask_works — smoke test на отсутствие NaN."
)
add_para(
    "test_without_gqa — если n_kv_heads не задан, он равен n_heads."
)
add_para(
    "test_residual_connection_not_broken — выход не нулевой и не равен "
    "входу. Проверка инициализации и работы проекций."
)
add_para(
    "test_different_lengths — attention работает для разной длины "
    "(5 и 20 токенов)."
)
add_para(
    "test_gqa_vs_vanilla_equivalence_single_token — для одного токена "
    "GQA и full MHA дают одинаковую форму."
)

# =============== 6 ===============
add_heading("6. Trade-offs и альтернативы", level=2)

add_para(
    "1. GQA (12:4) vs MHA (12:12): выбрана GQA ради экономии KV-кэша "
    "на CPU. Альтернатива MQA (12:1) — ещё больше экономия, но слишком "
    "грубая аппроксимация для 150M."
)
add_para(
    "2. bias=False: убирает ~300K параметров для d_model=768, 12 голов. "
    "Стандарт LLaMA, работает с RMSNorm."
)
add_para(
    "3. Xavier vs Kaiming: Xavier — стандарт для attention. Kaiming "
    "предпочтительнее для ReLU-слоёв, для attention разница незначительна."
)
add_para(
    "4. Маска генерируется на лету: можно было бы создать в __init__ "
    "и переиспользовать, но при разной длине маска разного размера. "
    "Генерация на лету — универсальный вариант."
)
add_para(
    "5. Все головы в один batch: вместо цикла for head используем "
    "reshape(-1, L, d_k). Разница в скорости — до 10x на GPU."
)

# =============== 7 ===============
add_heading("7. Связь с теоретической лекцией", level=2)

add_para(
    "Каждая строка кода прямо соотносится с формулами из Лекции 1:"
)

add_para(
    "- Формула Attention(Q,K,V)=softmax(Q*K^T/sqrt(d_k))*V реализована "
    "в ScaledDotProductAttention.forward.\n"
    "- Масштабирование sqrt(d_k) — нормировочный коэффициент из лекции.\n"
    "- Multi-head (строки 78-80) — параллельные attention-головы "
    "с формулой MultiHead(Q,K,V)=Concat(head_1,...,head_h)*W_o.\n"
    "- Каузальная маска — авторегрессионность decoder-only.\n"
    "- Обратная проекция W_o — Concat(...)*W_o в лекционной нотации."
)

add_para(
    "В текущей реализации отсутствует RoPE-позиционное кодирование — "
    "будет добавлено в следующем цикле как отдельный слой. "
    "Residual connections и LayerNorm — на уровне блока трансформера."
)

# =============== 8 ===============
add_heading("8. Производительность", level=2)

add_para(
    "Для модели 150M, d_model=768, 12 голов, d_k=64, контекст 2048:"
)

add_para(
    "- Q*K^T: O(B * L^2 * d_k) = 2 * 2048^2 * 64 ~= 537M операций на слой.\n"
    "- attn*V: O(B * L^2 * d_k) — те же 537M.\n"
    "- Итого на 12 слоёв: ~12.9 млрд операций на forward.\n"
    "- Скорость на CPU (4 ядра, ~10 GFLOPS): ~1-2 секунды на токен.\n"
    "- KV-кэш: 2 * L * n_kv_heads * d_k * sizeof(fp16) = "
    "2 * 2048 * 4 * 64 * 2 = 2 МБ на слой, 24 МБ на все 12 слоёв."
)

add_para(
    "Без GQA (n_kv_heads=12): 72 МБ — разница в 3 раза. "
    "Для CPU с 8GB RAM 50 МБ не критично, но для контекста 4096+ "
    "экономия становится существенной."
)

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "lecture-attention-implementation.docx")
doc.save(out_path)
print(f"Saved: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
