# Research Assistant — Week 1 Smoke Test Outputs

Generated from `scripts/week1_smoke.py` across 5 queries.
Total cost: $0.7565 · Total wallclock: 2429.5s


---

# Query 1 (VI): So sánh LoRA và QLoRA cho fine-tuning LLM năm 2026

# So sánh LoRA và QLoRA cho fine-tuning LLM năm 2026

> **Báo cáo nghiên cứu tự động** — sinh bởi Research Assistant Agent.
> *Thời gian tạo*: 2026-04-23T08:59:02+00:00 · *Chi phí*: $0.1499
> **Miễn trừ**: Nội dung do AI tổng hợp từ nguồn công khai, có thể chứa lỗi;
> người đọc cần kiểm chứng lại trước khi sử dụng trong quyết định quan trọng.
## Tóm lược kế hoạch
Câu hỏi gốc được phân rã thành 6 câu hỏi con:
1. **LoRA (Low-Rank Adaptation) là gì và nó hoạt động như thế nào trong fine-tuning các mô hình ngôn ngữ lớn (LLM)?**
   *Cần hiểu rõ khái niệm và cơ chế hoạt động của LoRA trước khi so sánh.*
2. **QLoRA (Quantized Low-Rank Adaptation) là gì và nó khác biệt với LoRA như thế nào về mặt kỹ thuật?**
   *Cần nắm được định nghĩa và đặc điểm kỹ thuật của QLoRA để có cơ sở so sánh.*
3. **So sánh hiệu suất và độ chính xác của LoRA và QLoRA trong fine-tuning LLM dựa trên các nghiên cứu và benchmark năm 2025-2026?**
   *Đánh giá hiệu quả thực tế của hai phương pháp về mặt chất lượng mô hình.*
4. **So sánh yêu cầu về bộ nhớ (memory) và tài nguyên tính toán giữa LoRA và QLoRA khi fine-tuning LLM?**
   *Hiểu rõ chi phí tài nguyên là yếu tố quan trọng khi lựa chọn phương pháp fine-tuning.*
5. **Các xu hướng và cải tiến mới nhất của LoRA và QLoRA trong năm 2025-2026 là gì?**
   *Cập nhật những phát triển gần đây để có cái nhìn toàn diện về trạng thái hiện tại.*
6. **Trong những trường hợp nào nên sử dụng LoRA và khi nào nên sử dụng QLoRA cho fine-tuning LLM theo khuyến nghị năm 2026?**
   *Tổng hợp các khuyến nghị thực tế để đưa ra hướng dẫn lựa chọn phù hợp.*

---

## 1. LoRA (Low-Rank Adaptation) là gì và nó hoạt động như thế nào trong fine-tuning các mô hình ngôn ngữ lớn (LLM)?

LoRA (Low-Rank Adaptation) là một kỹ thuật được thiết kế để giảm số lượng tham số cần tinh chỉnh trong quá trình fine-tuning các mô hình ngôn ngữ lớn (LLM) [^1]. Thay vì tinh chỉnh toàn bộ trọng số của mô hình, LoRA chỉ học một ma trận low-rank bổ sung trong các lớp (layers) trọng số gốc [^1]. Cách tiếp cận này giúp giảm đáng kể số lượng tham số huấn luyện, tiết kiệm bộ nhớ và tăng tốc độ huấn luyện [^1].

Về cơ chế hoạt động, LoRA đóng băng các trọng số của mô hình được đào tạo trước và chèn các ma trận phân rã rank vào từng lớp của kiến trúc Transformer [^3]. Ví dụ, so với GPT-3 175B được fine-tuning bằng Adam, LoRA có thể giảm số lượng tham số huấn luyện đi 10.000 lần và giảm yêu cầu bộ nhớ GPU đi 3 lần [^3]. Mặc dù có ít tham số huấn luyện hơn, LoRA vẫn đạt được chất lượng mô hình tương đương hoặc tốt hơn so với fine-tuning đầy đủ [^3].

Trong thực hành, LoRA được cấu hình thông qua các tham số chính: r (rank của ma trận low-rank), lora_alpha (hệ số scaling), lora_dropout (tỷ lệ dropout để tránh overfitting), và bias [^1][^2]. Các tham số này có thể được điều chỉnh tùy theo nhu cầu cụ thể của tác vụ. Sự ra đời của LoRA và các phương pháp tương tự đã dân chủ hóa quá trình fine-tuning, cho phép tinh chỉnh các mô hình khổng lồ trên các tài nguyên phần cứng hạn chế [^4].
## 2. QLoRA (Quantized Low-Rank Adaptation) là gì và nó khác biệt với LoRA như thế nào về mặt kỹ thuật?

QLoRA (Quantized Low-Rank Adaptation) là phiên bản cải tiến của LoRA được thiết kế để tinh chỉnh mô hình ngôn ngữ lớn (LLM) một cách hiệu quả hơn về bộ nhớ và tốc độ [^7]. Cả LoRA và QLoRA đều thuộc dạng parameter-efficient fine-tuning (PEFT), giúp tinh chỉnh LLM mà không cần cập nhật toàn bộ hàng tỷ trọng số truyền thống [^6].

Về mặt kỹ thuật, sự khác biệt chính giữa hai phương pháp nằm ở các kỹ thuật tối ưu hóa mà QLoRA áp dụng. Trong khi LoRA chỉ huấn luyện các ma trận có thứ hạng thấp (low-rank matrices) được thêm vào mô hình gốc [^6], QLoRA kết hợp cơ chế lượng tử hóa 4-bit với Low-Rank Adaptation [^7]. Ngoài ra, QLoRA còn ứng dụng các kỹ thuật tối ưu bổ sung như Double Quantization (lượng tử hóa kép) và Paged Optimizer nhằm giảm dung lượng lưu trữ hơn nữa mà vẫn giữ được hiệu suất mô hình [^7]. Những cải tiến này cho phép QLoRA hoạt động hiệu quả trên phần cứng tầm trung hoặc thấp, trong khi vẫn duy trì độ chính xác cao [^7].
## 3. So sánh hiệu suất và độ chính xác của LoRA và QLoRA trong fine-tuning LLM dựa trên các nghiên cứu và benchmark năm 2025-2026?

Chưa đủ dữ liệu để kết luận về so sánh hiệu suất và độ chính xác của LoRA và QLoRA dựa trên các nghiên cứu và benchmark năm 2025-2026. Các bằng chứng được cung cấp chủ yếu đến từ năm 2023 hoặc không có ngày công bố rõ ràng, không phải từ các nghiên cứu năm 2025-2026 như câu hỏi yêu cầu.

Tuy nhiên, dựa trên các nghiên cứu có sẵn, có thể nêu một số kết luận chung: QLoRA với kiểu dữ liệu NF4 (NormalFloat4) có khả năng đạt được hiệu suất tương đương với LoRA 16-bit truyền thống [^12]. Cụ thể, khi tinh chỉnh các mô hình LLaMA từ 7B đến 65B tham số trên các tập dữ liệu Alpaca và FLAN v2, QLoRA với NF4 đã hoàn toàn phục hồi hiệu suất MMLU của LoRA 16-bit [^12]. Ngược lại, QLoRA với FP4 lại kém hơn LoRA 16-bit khoảng 1 điểm phần trăm [^12].

Về mặt tài nguyên, cả LoRA và QLoRA đều là các kỹ thuật tinh chỉnh hiệu quả về tham số (PEFT), cho phép tiết kiệm bộ nhớ đáng kể bằng cách chỉ huấn luyện các ma trận có thứ hạng thấp thay vì cập nhật toàn bộ mô hình [^11][^13]. QLoRA thêm một lớp tối ưu hóa bằng cách lượng tử hóa mô hình gốc xuống 4-bit, từ đó giảm yêu cầu tài nguyên hơn nữa so với LoRA [^11][^13].
## 4. So sánh yêu cầu về bộ nhớ (memory) và tài nguyên tính toán giữa LoRA và QLoRA khi fine-tuning LLM?

QLoRA được thiết kế như một phiên bản cải tiến của LoRA nhằm giảm yêu cầu về bộ nhớ khi fine-tuning LLM. QLoRA sử dụng lượng tử hóa 4-bit kết hợp với Low-Rank Adaptation, cho phép tối ưu hóa bộ nhớ hiệu quả hơn so với LoRA truyền thống [^16]. Cụ thể, QLoRA áp dụng các kỹ thuật tối ưu bổ sung như Double Quantization (lượng tử hóa kép) và Paged Optimizer để giảm dung lượng lưu trữ thêm nữa [^16].

Về chi tiết kỹ thuật, trong LoRA, phần lớn bộ nhớ được sử dụng không phải từ các tham số LoRA mà từ activation gradients. Ví dụ, với mô hình 7B LLaMA, các LoRA input gradients chiếm 567 MB bộ nhớ trong khi các LoRA parameters chỉ chiếm 26 MB [^17]. Ngược lại, QLoRA sử dụng một kiểu dữ liệu lưu trữ độ chính xác thấp (thường là 4-bit) và một kiểu dữ liệu tính toán là BFloat16, giúp giảm đáng kể yêu cầu về bộ nhớ [^17]. QLoRA được đặc biệt thiết kế để hoạt động hiệu quả trên GPU tầm trung hoặc thấp [^16].

Tuy nhiên, dữ liệu cung cấp không chứa so sánh chi tiết về yêu cầu tài nguyên tính toán (như tốc độ xử lý, thời gian training) giữa hai phương pháp này. Chỉ có thông tin rằng QLoRA được thiết kế để tinh chỉnh "hiệu quả hơn về bộ nhớ và tốc độ" [^16], nhưng không có số liệu cụ thể để so sánh chi tiết.
## 5. Các xu hướng và cải tiến mới nhất của LoRA và QLoRA trong năm 2025-2026 là gì?

Chưa đủ dữ liệu để kết luận về các xu hướng và cải tiến mới nhất của LoRA và QLoRA trong năm 2025-2026. Các bằng chứng được cung cấp chỉ chứa thông tin từ năm 2023 và các tài liệu không cụ thể về dự báo phát triển trong giai đoạn 2025-2026. Tài liệu [^21] chỉ đề cập đến việc cập nhật dữ liệu pháp luật cho năm 2024-2025 trong một hệ thống trợ lý ảo, không liên quan đến xu hướng của LoRA/QLoRA. Tài liệu [^22] là một bài báo khoa học từ tháng 5 năm 2023 về QLoRA, nhưng nó không chứa thông tin về các cải tiến dự kiến cho năm 2025-2026.
## 6. Trong những trường hợp nào nên sử dụng LoRA và khi nào nên sử dụng QLoRA cho fine-tuning LLM theo khuyến nghị năm 2026?

Chưa đủ dữ liệu để kết luận về khuyến nghị năm 2026 về việc sử dụng LoRA và QLoRA cho fine-tuning LLM. Mặc dù bằng chứng [^23] cung cấp thông tin chi tiết về QLoRA và các chi tiết thử nghiệm so sánh giữa QLoRA và LoRA, nhưng tài liệu này được xuất bản vào năm 2023 và không chứa bất kỳ khuyến nghị cụ thể nào cho năm 2026. Các bằng chứng khác [^24][^25][^26][^27] không liên quan đến chủ đề fine-tuning LLM với LoRA/QLoRA mà thay vào đó đề cập đến các lĩnh vực hoàn toàn khác như phong thủy xây nhà, ứng dụng AI trong doanh nghiệp, xét nghiệm y tế, và lập trình TypeScript.

---

## Tài liệu tham khảo
[^1]: [Thực chiến Fine-Tuning mô hình ngôn ngữ lớn Bloom-560m - AIcandy](https://aicandy.vn/thuc-chien-fine-tuning-mo-hinh-ngon-ngu-lon-bloom-560m/)[^2]: [Thực chiến Fine-Tuning mô hình ngôn ngữ lớn Phi-2 - AIcandy](https://aicandy.vn/thuc-chien-fine-tuning-mo-hinh-ngon-ngu-lon-phi-2)[^3]: [LoRA: Low-Rank Adaptation of Large Language Models](http://arxiv.org/abs/2106.09685v2) — 2021-06-17[^4]: [Giải mã LLM Fine-tuning: Cách "cá nhân hóa" mô hình ngôn ngữ lớn cho nhu cầu riêng của Bạn](https://tinai.vn/kien-thuc-ai/giai-ma-llm-fine-tuning-cach-ca-nhan-hoa-mo-hinh-ngon-ngu-lon-cho-nhu-cau-rieng-cua-ban.html)[^5]: [Fine-tuning LLM: Điều chỉnh các Mô hình Ngôn ngữ Lớn cho các yêu cầu riêng - MyGPT](https://mygpt.vn/fine-tuning-llm-dieu-chinh-cac-mo-hinh-ngon-ngu-lon-cho-cac-yeu-cau-rieng)[^6]: [LoRA vs. QLoRA – Tinh chỉnh LLM hiệu quả và tiết kiệm - HBLAB JSC](https://hblab.vn/lora-vs-qlora-tinh-chinh-llm-hieu-qua-va-tiet-kiem/)[^7]: [QLoRA là gì? Giải pháp tối ưu bộ nhớ và tinh chỉnh LLM hiệu quả](https://vnptai.io/vi/blog/detail/qlora-la-gi)[^8]: [LoRA: Low-Rank Adaptation of Large Language Models](http://arxiv.org/abs/2106.09685v2) — 2021-06-17[^9]: [QLoRA: Efficient Finetuning of Quantized LLMs](http://arxiv.org/abs/2305.14314v1) — 2023-05-23[^10]: [Ollama là gì? Cách sử dụng Ollama](https://apidog.com/vi/blog/how-to-use-ollama-vi/)[^11]: [LoRA vs. QLoRA – Tinh chỉnh LLM hiệu quả và tiết kiệm - HBLAB JSC](https://hblab.vn/lora-vs-qlora-tinh-chinh-llm-hieu-qua-va-tiet-kiem)[^12]: [QLoRA: Efficient Finetuning of Quantized LLMs](http://arxiv.org/abs/2305.14314v1) — 2023-05-23[^13]: [Tutorial: Low-rank Adaptation Techniques in Fine-tuning a Large Language Model](https://aivietnam.edu.vn/blog/finetune-lora-llms)[^14]: [Fine-tuning Là Gì? So Sánh Fine-tuning Và Pre-Training](https://fpt.ai/vi/bai-viet/fine-tuning)[^15]: [BÀI SOẠN VỀ SIÊU ÂM CHẨN ĐOÁN: Nghiên cứu So sánh giữa LLM và Chuyên gia trong việc Hỗ trợ Siêu âm : Thách thức và Hạn chế](https://www.nguyenthienhung.com/2025/12/so-sanh-cac-ngon-ngu-lon-va-chuyen-gia.html)[^16]: [QLoRA là gì? Giải pháp tối ưu bộ nhớ và tinh chỉnh LLM hiệu quả - VNPT AI](https://vnptai.io/vi/blog/detail/qlora-la-gi)[^17]: [QLoRA: Efficient Finetuning of Quantized LLMs](http://arxiv.org/abs/2305.14314v1) — 2023-05-23[^18]: [Xây dựng Chatbot được cá nhân hóa: LSTM so với LLM được tinh ...](https://vn.linkedin.com/pulse/building-personalized-chatbot-lstm-vs-fine-tuned-llm-gpt-2-baksi-auiec?tl=vi)[^19]: [AI VIET NAM - Facebook](https://www.facebook.com/groups/aivietnam.edu.vn/posts/2447229765735316/)[^20]: [Giải mã LLM Fine-tuning: Cách "cá nhân hóa" mô hình ngôn ngữ ...](https://tinai.vn/kien-thuc-ai/giai-ma-llm-fine-tuning-cach-ca-nhan-hoa-mo-hinh-ngon-ngu-lon-cho-nhu-cau-rieng-cua-ban.html)[^21]: [Đề Tài: Xây Dựng Hệ Thống Trợ Lý Ảo Tư Vấn Pháp Luật Việt Nam ...](https://www.studocu.vn/vn/document/university-of-economics-hcmc-international-school-of-business/random/de-tai-xay-dung-he-thong-tro-ly-ao-tu-van-phap-luat-viet-nam-dua-tren-llm-va-ky/150671515)[^22]: [QLoRA: Efficient Finetuning of Quantized LLMs](http://arxiv.org/abs/2305.14314v1) — 2023-05-23[^23]: [QLoRA: Efficient Finetuning of Quantized LLMs](http://arxiv.org/abs/2305.14314v1) — 2023-05-23[^24]: [Năm 2026 xây nhà, chọn hướng nào cho mát, dễ ở và bền vượng?](https://vietnamnet.vn/nam-2026-xay-nha-chon-huong-nao-cho-mat-de-o-va-ben-vuong-2484849.html)[^25]: [Khi nào nên dùng AI và khi nào không nên sử dụng? | IRTECH](https://irtech.com.vn/khi-nao-nen-dung-ai-va-khi-nao-khong-nen-su-dung/)[^26]: [Khi nào nên sử dụng test nhanh HIV và test nhanh ma túy](https://www.webtretho.vn/f/dia-chi-kham-chua-benh/khi-nao-nen-su-dung-test-nhanh-hiv-va-test-nhanh-ma-tuy)[^27]: [Khi nào nên sử dụng interfaces và khi nào nên sử dụng classes trong TypeScript](https://kungfutech.edu.vn/cau-hoi-phong-van/khi-nao-nen-su-dung-interfaces-va-khi-nao-nen-su-dung-classes-trong-typescript)
---
*Báo cáo này được sinh tự động; các trích dẫn `[^N]` tương ứng với thứ tự
tài liệu ở mục Tài liệu tham khảo.*
*Trace đầy đủ trên Langfuse*: [https://cloud.langfuse.com/project/cmo85r1z800cnad07chmi8dt8/traces/1958cda9727acd39d61a065b9e2825e1](https://cloud.langfuse.com/project/cmo85r1z800cnad07chmi8dt8/traces/1958cda9727acd39d61a065b9e2825e1)


---

# Query 2 (VI): Retrieval-Augmented Generation là gì, khi nào nên dùng thay vì fine-tuning?

# Retrieval-Augmented Generation là gì, khi nào nên dùng thay vì fine-tuning?

> **Báo cáo nghiên cứu tự động** — sinh bởi Research Assistant Agent.
> *Thời gian tạo*: 2026-04-23T09:10:05+00:00 · *Chi phí*: $0.1886
> **Miễn trừ**: Nội dung do AI tổng hợp từ nguồn công khai, có thể chứa lỗi;
> người đọc cần kiểm chứng lại trước khi sử dụng trong quyết định quan trọng.
## Tóm lược kế hoạch
Câu hỏi gốc được phân rã thành 5 câu hỏi con:
1. **Retrieval-Augmented Generation (RAG) là gì và nó hoạt động như thế nào?**
   *Need to establish foundational understanding of RAG before comparing it with alternatives.*
2. **Fine-tuning trong machine learning và large language models là gì?**
   *Must understand fine-tuning to make meaningful comparisons with RAG.*
3. **Ưu điểm và nhược điểm của Retrieval-Augmented Generation là gì?**
   *Understanding RAG's strengths and weaknesses helps determine when to use it.*
4. **Ưu điểm và nhược điểm của fine-tuning large language models là gì?**
   *Understanding fine-tuning's strengths and weaknesses enables comparison with RAG.*
5. **Trong những trường hợp nào nên sử dụng Retrieval-Augmented Generation thay vì fine-tuning?**
   *Directly addresses the core comparison question about when to choose RAG over fine-tuning.*

---

## 1. Retrieval-Augmented Generation (RAG) là gì và nó hoạt động như thế nào?

Retrieval-Augmented Generation (RAG) là một phương pháp trong trí tuệ nhân tạo kết hợp giữa hai thành phần chính: mô hình ngôn ngữ lớn (LLM) và khả năng truy xuất thông tin từ các nguồn bên ngoài [^1][^2]. Cụ thể, RAG là một kiến trúc hệ thống cho phép LLM truy cập và tham chiếu đến các nguồn tài liệu bên ngoài (Knowledge Base) trước khi tạo ra câu trả lời [^4].

RAG hoạt động bằng cách kết hợp giữa truy xuất thông tin và mô-đun sinh nội dung để tạo ra các phản hồi tự nhiên, chính xác [^2]. Phương pháp này giải quyết những giới hạn của các mô hình ngôn ngữ lớn truyền thống, vốn đôi khi tạo ra những câu trả lời nghe có vẻ hợp lý nhưng thực chất là sai sự thật vì chúng chỉ dựa vào dữ liệu đã được huấn luyện [^5]. Nhờ RAG, AI có thể truy cập dữ liệu thực chứng từ bên ngoài, giúp thoát khỏi giới hạn của dữ liệu huấn luyện cũ kỹ và tạo ra những câu trả lời cập nhật, chính xác và phù hợp hơn [^4][^5].
## 2. Fine-tuning trong machine learning và large language models là gì?

Fine-tuning trong machine learning là quá trình tiếp tục huấn luyện một mô hình đã được huấn luyện trước đó trên một tập dữ liệu mới, nhỏ hơn và cụ thể cho một nhiệm vụ nhất định, nhằm cải thiện hiệu suất của mô hình mà không cần phải huấn luyện từ đầu [^6]. Quá trình này liên quan đến việc điều chỉnh các tham số của mô hình thông qua lan truyền ngược (backpropagation) trên tập dữ liệu cụ thể cho từng nhiệm vụ [^6].

Đối với các mô hình ngôn ngữ lớn (LLM), fine-tuning là một kỹ thuật mạnh mẽ để thích ứng các mô hình được huấn luyện trước với các nhiệm vụ và lĩnh vực cụ thể [^6][^8]. Mặc dù các mô hình được huấn luyện trước như GPT-3 có thể hoạt động tốt trong các nhiệm vụ chung, nhưng chúng thường không thể vượt trội hơn một mô hình được fine-tune trong các vai trò tập trung hơn [^8]. Có hai phương pháp fine-tuning chính cho LLM: Supervised Learning (học có giám sát) và Reinforcement Learning from Human Feedback (RLHF - học tăng cường từ phản hồi của con người) [^10]. Trong khi Supervised Learning sử dụng các cặp đầu vào-đầu ra được gắn nhãn, RLHF sử dụng phản hồi từ con người làm tín hiệu phần thưởng để căn chỉnh hành vi của mô hình với các sở thích của con người [^10].

Một thách thức quan trọng của fine-tuning là "catastrophic forgetting" (quên mất kiến thức), nơi mô hình có thể mất đi những kiến thức quý báu từ quá trình huấn luyện trước khi học một nhiệm vụ mới [^6]. Để giải quyết vấn đề này, các chuyên gia đang chuyển sang Reinforcement Fine-Tuning (RFT), một kỹ thuật kết hợp các hệ thống phần thưởng với dữ liệu huấn luyện được nhắm mục tiêu, cung cấp một cách tiếp cận động hơn so với học có giám sát truyền thống [^7].
## 3. Ưu điểm và nhược điểm của Retrieval-Augmented Generation là gì?

**Ưu điểm của Retrieval-Augmented Generation**

Retrieval-Augmented Generation (RAG) là một khung AI giúp nâng cao các mô hình ngôn ngữ lớn bằng cách kết nối chúng với các nguồn thông tin đáng tin cậy [^14]. Một ưu điểm quan trọng của RAG là giảm thiểu các vấn đề về độ chính xác thực tế - các mô hình ngôn ngữ lớn thường tạo ra các phản hồi chứa những sai lệch thực tế do chỉ dựa vào kiến thức tham số mà chúng chứa đựng, nhưng RAG giảm được những vấn đề này [^15]. Ngoài ra, RAG nâng cao chất lượng và tính xác thực của mô hình ngôn ngữ thông qua việc truy xuất thông tin và tự phản ánh [^15].

**Nhược điểm của Retrieval-Augmented Generation**

Nhược điểm chính của RAG là tốn kém, chậm và không theo thời gian thực [^11]. Những hạn chế này có thể ảnh hưởng đến khả năng triển khai RAG trong các ứng dụng yêu cầu xử lý nhanh chóng hoặc cập nhật thông tin liên tục.
## 4. Ưu điểm và nhược điểm của fine-tuning large language models là gì?

Dựa trên các bằng chứng được cung cấp, tôi có thể trích xuất một số ưu điểm của fine-tuning large language models, mặc dù bằng chứng về nhược điểm là hạn chế.

Về ưu điểm, fine-tuning cho phép đạt được những cải thiện đáng kể về hiệu suất thông qua các kỹ thuật như RLHF (Reinforcement Learning from Human Feedback)[^16]. Fine-tuning cũng rất hữu ích trong các bối cảnh chuyên biệt như healthcare, legal practice và các lĩnh vực khác có thuật ngữ độc quyền, nơi mà việc điều chỉnh mô hình embedding trên dữ liệu miền cụ thể của bạn trở nên cần thiết để giảm thiểu sự khác biệt[^18]. Ngoài ra, fine-tuning có thể được sử dụng để căn chỉnh retriever và generator, chẳng hạn như sử dụng kết quả từ LLM làm tín hiệu giám sát cho quá trình fine-tuning[^18]. Instruction tuning cũng cho phép đạt được hiệu suất zero-shot trên các tác vụ chưa được nhìn thấy[^16].

Tuy nhiên, bằng chứng được cung cấp không chứa thông tin rõ ràng về các nhược điểm của fine-tuning large language models. Các tài liệu chỉ đề cập rằng việc căn chỉnh với sở thích con người là một quá trình "vẫn đang được khám phá và tinh chỉnh"[^16], nhưng không cung cấp chi tiết cụ thể về những thách thức hoặc hạn chế của fine-tuning. Do đó, tôi không thể cung cấp một danh sách toàn diện về nhược điểm dựa trên bằng chứng hiện có.
## 5. Trong những trường hợp nào nên sử dụng Retrieval-Augmented Generation thay vì fine-tuning?

RAG nên được sử dụng khi mục tiêu chính là cung cấp kiến thức thực tế và thông tin chính xác, trong khi fine-tuning phù hợp hơn khi cần thay đổi hành vi hoặc phong cách của mô hình [^25]. Cụ thể, RAG là lựa chọn tối ưu khi bạn cần truy cập vào thông tin không có trong kiến thức ban đầu của mô hình LLM, chẳng hạn như dữ liệu cập nhật hoặc thông tin chuyên biệt [^21]. RAG cũng xuất sắc trong việc xử lý các tác vụ yêu cầu kiến thức sâu rộng bằng cách tận dụng các tài liệu bên ngoài (như Wikipedia) để tăng cường khả năng của mô hình [^22].

Một trường hợp sử dụng quan trọng của RAG là khi bạn cần truy cập dữ liệu độc quyền hoặc nội bộ của công ty [^25]. Tuy nhiên, RAG yêu cầu kết nối internet hoặc truy cập cơ sở dữ liệu, do đó trong các tình huống offline hoặc không có kết nối mạng, fine-tuning sẽ là lựa chọn phù hợp hơn [^25]. Ngược lại, fine-tuning nên được ưu tiên khi bạn muốn điều chỉnh phong cách trả lời, sử dụng thuật ngữ riêng của ngành/công ty, hoặc định dạng đầu ra theo cấu trúc cụ thể [^25]. Thực tế, hai phương pháp này có thể kết hợp: bạn có thể fine-tune mô hình để đạt được phong cách mong muốn, sau đó sử dụng nó trong quy trình RAG để truy xuất và kết hợp thông tin cập nhật từ cơ sở dữ liệu [^25].

---

## Tài liệu tham khảo
[^1]: [Retrieval-augmented generation (RAG) là gì? - VNPT AI](https://vnptai.io/vi/blog/detail/rag-la-gi)[^2]: [RAG (Retrieval Augmented Generation) là gì? Mô hình RAG ...](https://lacviet.vn/retrieval-augmented-generation/)[^3]: [RAG là gì? & AI Search thực sự hoạt động theo cách nào?](https://www.brandsvietnam.com/congdong/topic/rag-la-gi-ai-search-thuc-su-hoat-dong-theo-cach-nao)[^4]: [RAG (Retrieval-Augmented Generation): Giải Pháp Đột Phá Xây Dựng Trí Tuệ Nhân Tạo Đáng Tin Cậy Cho Doanh Nghiệp - INDA - Insight Data](https://inda.vn/rag-retrieval-augmented-generation/)[^5]: [Retrieval-Augmented Generation là gì? Tác động của RAG đến tìm kiếm và TMĐT - SEONGON](https://seongon.com/blog/ai/retrieval-augmented-generation-la-gi.html)[^6]: [Fine-Tuning Large Language Models - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2023/08/fine-tuning-large-language-models?ref=global_footer)[^7]: [Reinforcement Fine-Tuning (RFT) For Large Language Models: Safer AI?](https://youraitips.com/reinforcement-fine-tuning-rft-for-large-language-models-safer-ai)[^8]: [Advanced Techniques for Fine Tuning Large Language Models in 2024](https://pixldata.com/blog/advanced-techniques-for-fine-tuning-large-language-models-in-2024/)[^9]: [Fine-Tuning Large Language Models - The Basics with HuggingFace - Corpnce](https://www.corpnce.com/fine-tuning-large-language-models-the-basics-with-huggingface)[^10]: [Fine-Tuning Large Language Models: Supervised Learning and RLHF Techniques Compared](https://www.linkedin.com/pulse/fine-tuning-large-language-models-supervised-learning-bandiatmakur-ox7ic)[^11]: [Cách các mô hình ngôn ngữ lớn học, cập nhật và xếp hạng nội dung](https://vn.linkedin.com/pulse/how-large-language-models-learn-update-rank-content-gordon-orgxe?tl=vi)[^12]: [Apa itu Vector Database? Ini Penjelasan Gampang Buat Pemula ...](https://www.tiktok.com/@typeclickdone/video/7516199638735621394)[^13]: [Cos'è un Vector Database? Spiegazione Semplice - TikTok](https://www.tiktok.com/@claudio.hypergrowth/video/7560374888482016534)[^14]: [RAG đang nổi lên như một mắt xích còn thiếu giữa sự cường điệu ...](https://vn.linkedin.com/pulse/rag-emerging-missing-link-between-ai-hype-real-world-wajeeh-ul-hassan-dxm7f?tl=vi)[^15]: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](http://arxiv.org/abs/2310.11511v1) — 2023-10-17[^16]: [Llama 2: Open Foundation and Fine-Tuned Chat Models](http://arxiv.org/abs/2307.09288v2) — 2023-07-18[^17]: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](http://arxiv.org/abs/2310.11511v1) — 2023-10-17[^18]: [Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997v5) — 2023-12-18[^19]: [ưu điểm và nhược điểm Tiếng Anh là gì](https://tudien.dolenglish.vn/uu-diem-va-nhuoc-diem-tieng-anh-la-gi)[^20]: [Ưu và nhược điểm của Memoization hoặc phương pháp TopDown là gì](https://kungfutech.edu.vn/cau-hoi-phong-van/uu-va-nhuoc-diem-cua-memoization-hoac-phuong-phap-top-down-la-gi)[^21]: [Retrieval-Augmented Generation: Phương pháp không thể thiếu khi triển khai các dự án LLM trong thực tế! (Phần 1)](https://viblo.asia/p/retrieval-augmented-generation-phuong-phap-khong-the-thieu-khi-trien-khai-cac-du-an-llm-trong-thuc-te-phan-1-Ny0VG7yzVPA)[^22]: [Retrieval Augmented Generation with Huggingface Transformers and Ray](https://huggingface.co/blog/ray-rag) — 2021-02-10[^23]: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](http://arxiv.org/abs/2310.11511v1) — 2023-10-17[^24]: [Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997v5) — 2023-12-18[^25]: [RAG là gì và khi nào nên sử dụng nó thay vì Fine-Tuning? - Evotek Careers](https://tuyendung.evotek.vn/ai-engineer-roadmap-rag-la-gi-va-khi-nao-nen-su-dung-no-thay-vi-fine-tuning)
---
*Báo cáo này được sinh tự động; các trích dẫn `[^N]` tương ứng với thứ tự
tài liệu ở mục Tài liệu tham khảo.*
*Trace đầy đủ trên Langfuse*: [https://cloud.langfuse.com/project/cmo85r1z800cnad07chmi8dt8/traces/0ec0502508587f3044ce0a165a3f7de4](https://cloud.langfuse.com/project/cmo85r1z800cnad07chmi8dt8/traces/0ec0502508587f3044ce0a165a3f7de4)


---

# Query 3 (EN): What are the latest advances in reasoning models like OpenAI o3 and DeepSeek R1 in 2026?

# What are the latest advances in reasoning models like OpenAI o3 and DeepSeek R1 in 2026?

> **Automated research report** — generated by Research Assistant Agent.
> *Generated at*: 2026-04-23T09:17:35+00:00 · *Cost*: $0.1387
> **Disclaimer**: AI-synthesized from public sources. Readers should verify
> before relying on this for high-stakes decisions.
## Plan overview
The original question was decomposed into 6 sub-questions:
1. **What is OpenAI o3 and what are its key features and capabilities as a reasoning model?**
   *Understanding the fundamental architecture and capabilities of OpenAI o3 provides essential context for evaluating its advances.*
2. **What is DeepSeek R1 and what are its key features and capabilities as a reasoning model?**
   *Understanding the fundamental architecture and capabilities of DeepSeek R1 provides essential context for evaluating its advances.*
3. **What are the latest advances and improvements in OpenAI o3 announced or released in 2025-2026?**
   *This captures the most recent developments specific to OpenAI o3 within the timeframe of interest.*
4. **What are the latest advances and improvements in DeepSeek R1 announced or released in 2025-2026?**
   *This captures the most recent developments specific to DeepSeek R1 within the timeframe of interest.*
5. **How do OpenAI o3 and DeepSeek R1 compare in terms of reasoning performance, benchmarks, and capabilities in 2026?**
   *A comparative analysis helps contextualize the relative strengths and advances of both models.*
6. **What are the broader trends and breakthroughs in AI reasoning models in 2025-2026 beyond OpenAI o3 and DeepSeek R1?**
   *Understanding the wider landscape helps position these specific models within the context of general progress in reasoning AI.*

---

## 1. What is OpenAI o3 and what are its key features and capabilities as a reasoning model?

OpenAI o3 is an advanced reasoning model developed by OpenAI, designed to push the boundaries of reasoning, answer reliability, and broad generalist capabilities.[^2] It is the successor to o1-preview and represents a major milestone in AI development.[^1] The o3 model family is specifically tailored for step-by-step logical problem-solving, allowing the model to effectively "think" through tasks and ensure more reliable and accurate outputs in areas like mathematics, science, and complex decision-making.[^5]

Key features of OpenAI o3 include a 200K token context window (significantly larger than its predecessor o1-preview's 128K),[^1] advanced reasoning capabilities, text and image processing, and autonomous tool use including web browsing, Python, file handling, and image generation.[^1] The model is optimized for coding and STEM tasks and introduces a novel technique called "deliberative alignment," aimed at aligning the model's reasoning capabilities with OpenAI's safety principles.[^5] o3 excels in critical reasoning and factual analysis, making it particularly strong for analytical and research tasks.[^2]

In terms of performance, o3 demonstrates exceptional benchmark results: it scored 87.7% on the GPQA-Diamond Benchmark, significantly outperforming other models including OpenAI o1 (76.0%) and DeepSeek R1 (71.5%).[^4] The model also achieved a leading Codeforces Elo rating of 2727 and 96.7% accuracy on the AIME test.[^4] Typical use cases include complex problem solving, advanced coding tasks, scientific research, multimodal applications, and AI agent development.[^1]
## 2. What is DeepSeek R1 and what are its key features and capabilities as a reasoning model?

DeepSeek-R1 is an open-source language model developed by Chinese AI startup DeepSeek, founded in 2023 by Liang Wenfeng [^10]. It is designed to perform text-based tasks similar to other advanced AI models while offering greater affordability and accessibility [^10]. The model is built on a large language model (LLM) architecture trained on a massive corpus of data [^10].

A key distinguishing feature of DeepSeek-R1 is its training methodology. Unlike conventional models that rely primarily on supervised fine-tuning, DeepSeek-R1 employs reinforcement learning directly to train its reasoning skills [^8]. The model comes in multiple variants: DeepSeek-R1-Zero, which is trained entirely through reinforcement learning and focuses on raw reasoning capabilities, and DeepSeek-R1 (Hybrid), which combines reinforcement learning with cold-start data from human-curated chain-of-thought examples to balance reasoning accuracy and readability [^9].

In terms of capabilities, DeepSeek-R1 demonstrates exceptional performance in mathematics and coding [^6], and is designed to enhance natural language processing (NLP), code generation, and various other AI-driven tasks [^7]. The model also features multi-agent learning capabilities that support coordinated decision-making in complex environments such as logistics, autonomous vehicles, and multi-player gaming [^9]. Additionally, pre-trained versions can be deployed for common tasks like recommendation systems, predictive analytics, and chatbots [^9].
## 3. What are the latest advances and improvements in OpenAI o3 announced or released in 2025-2026?

Based on the available evidence, the only specific o3 model advance announced in 2025–2026 is the release of o3-mini. OpenAI introduced o3-mini on January 31, 2025, described as a cost-efficient reasoning model optimized for coding, math, and related tasks [^11]. However, the evidence snippet does not provide detailed information about the specific improvements or capabilities of o3-mini beyond this basic characterization.

Regarding the full o3 model, the evidence does not clarify whether a complete o3 release occurred during 2025–2026 or provide any advances specific to it. The evidence instead mentions GPT-5.4 as a separate model line launched by OpenAI in March 2026 [^12], which is distinct from the o3 family.

While evidence [5] references OpenAI's 2026 roadmap, the provided snippet does not contain substantive details about o3 or o3-mini developments. Insufficient evidence is available in the provided sources to comprehensively describe the latest advances and improvements in OpenAI o3 announced or released during 2025–2026 beyond the basic announcement of o3-mini's release date and general purpose.
## 4. What are the latest advances and improvements in DeepSeek R1 announced or released in 2025-2026?

Based on the available evidence, DeepSeek R1's initial release in January 2025 introduced several technical advances, but there is limited documentation of subsequent improvements during 2025-2026.

At its January 2025 release, DeepSeek R1 featured a reinforcement learning-based training methodology with specific technical optimizations: a reduced temperature of 0.7 to prevent incoherent generation, 1,700 total training steps, and incorporation of preference-based rewards in the final 400 steps to avoid reward hacking [^17]. The model was evaluated across comprehensive benchmarks including MMLU, GPQA Diamond, SWE-Bench Verified, LiveCodeBench, AIME 2024, and Codeforces [^17]. Performance improvements between R1-Zero and R1 Dev1 were substantial, particularly in instruction-following as demonstrated by higher IF-Eval and ArenaHard scores [^17]. Competitively, R1 matched GPT-4 on most benchmarks [^16].

Regarding post-release developments during 2025-2026, the evidence is insufficient to identify specific advances or improvements to R1 itself. While evidence mentions DeepSeek V3 being announced in December 2025 [^18] and references to R1 in April 2026 reviews [^19], these do not describe technical improvements to R1 specifically. The available evidence documents R1's capabilities at its January 2025 release but does not provide substantive information about subsequent updates or enhancements to the R1 model line during the remainder of 2025 and into 2026.
## 5. How do OpenAI o3 and DeepSeek R1 compare in terms of reasoning performance, benchmarks, and capabilities in 2026?

# Comparison of OpenAI o3 and DeepSeek R1 in 2026

**Reasoning Performance and Math Benchmarks:**
OpenAI o3 demonstrates superior performance on mathematical reasoning tasks, achieving 99.2% on the MATH-500 benchmark, while DeepSeek R1 scores 97.3% on the same benchmark [^21]. This represents a narrow single-digit percentage gap between the two platforms [^21]. However, there is a discrepancy in the evidence regarding DeepSeek R1's exact math performance: one source reports 96% accuracy on math tasks compared to 97% for OpenAI's o1 and o3 models [^24]. Despite this minor variance in reported scores, DeepSeek R1 is characterized as delivering "top-tier reasoning across math, coding, and complex logic tasks on par with OpenAI's o1" [^22], and one expert assessment notes that "DeepSeek R1 gives you 95% of o3's math capability at 20% of the cost" [^21].

**Broader Capabilities and Practical Differences:**
Beyond pure math benchmarks, the models differ in their architectural approaches and deployment models. DeepSeek R1 uses a 671B parameter model with only 37B activated, enabling resource efficiency while maintaining advanced reasoning capabilities [^22]. OpenAI's o3 is positioned as part of a broader ecosystem emphasizing "frontier model capabilities (e.g., GPT‑4.1's 1 million‑token context window, advanced coding, and multimodal skills)" alongside practical considerations like latency and cost-effectiveness [^22]. DeepSeek R1 excels specifically in "complex mathematical reasoning" [^23], while o3-mini is noted as "arguably the safest for broad consumer use" [^23]. The pricing gap is substantial, with o3 costing approximately 18x more on input tokens than DeepSeek R1 [^21].
## 6. What are the broader trends and breakthroughs in AI reasoning models in 2025-2026 beyond OpenAI o3 and DeepSeek R1?

*(No synthesized answer available for this sub-question.)*

---

## References
[^1]: [OpenAI o3 (200k) - LLM Model - TokenCalculator.com](https://tokencalculator.com/model/o3)[^2]: [OpenAI o3 Model – Features, Pricing, Comparison | GlobalGPT](https://www.glbgpt.com/sitepage/openai-o3)[^3]: [OpenAI to Launch o3 Mini Reasoning AI Model Soon, What Should We Know?](https://diringkas.com/en/openai-soon-to-launch-o3-mini-reasoning-ai-model-what-we-need-to-know)[^4]: [OpenAI o3: Release Date, Features and Model Comparison](https://www.analyticsvidhya.com/blog/2025/01/openai-o3-vs-competitors-performance-and-applications/)[^5]: [OpenAI Launches O3 AI Model Family with Advanced Reasoning](https://tecknexus.com/openai-launches-o3-ai-model-family-with-advanced-reasoning)[^6]: [DeepSeek-R1 Part 1: Opportunities and Enterprise Risks of Open ...](https://www.allganize.ai/en/blog/the-emergence-of-deepseek-r1-and-what-we-must-not-overlook---part-1)[^7]: [DeepSeek R1 Explained: A Comprehensive Guide](https://www.bombaysoftwares.com/blog/deepseek-r1-comprehensive-guide)[^8]: [DeepSeek: Facts, Not Hype](https://www.netguru.com/blog/deepseek-r1-facts-not-hype)[^9]: [Introduction to DeepSeek R-1 Model - GeeksforGeeks](https://www.geeksforgeeks.org/techtips/introduction-to-deepseek-r1-models/)[^10]: [What Is DeepSeek-R1? | Built In](https://builtin.com/artificial-intelligence/deepseek-r1)[^11]: [Model Release Notes - OpenAI Help Center](https://help.openai.com/en/articles/9624314-model-release-notes)[^12]: [March 2026 Global AI Industry Highlights Recap - U深搜 - UniFuncs](https://unifuncs.com/s/VirUXXoQ)[^13]: [Gen AI for Business #90 "Happy New Year!" - LinkedIn](https://www.linkedin.com/pulse/gen-ai-business-90-happy-new-year-eugina-jordan-spqge)[^14]: [[PDF] How China's Open AI Strategy Reinforces Its Industrial Dominance](https://www.uscc.gov/sites/default/files/2026-03/Two_Loops--How_Chinas_Open_AI_Strategy_Reinforces_Its_Industrial_Dominance.pdf)[^15]: [OpenAI's 2026 Roadmap: From Chatbot to AI Super‑Assistant ...](https://medium.com/towards-explainable-ai/openais-2026-roadmap-from-chatbot-to-ai-super-assistant-disrupting-everything-f28b3754ddad)[^16]: [Why DeepSeek will outlast OpenAI, and it has nothing to do with AI ...](https://www.instagram.com/reel/DTl-TrpDoTd/)[^17]: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](http://arxiv.org/abs/2501.12948v2) — 2025-01-22[^18]: [LLMs 2025 Report: Progress, Problems, and Predictions - LinkedIn](https://www.linkedin.com/posts/sebastianraschka_i-just-uploaded-my-state-of-llms-2025-report-activity-7411781706778595328-IXVQ)[^19]: [April 2026 AI Models: Every Major Release Reviewed - Medium](https://medium.com/@sanjeevpatel3007/april-2026-ai-models-every-major-release-reviewed-6ea03d7bc0b7)[^20]: [DeepSeek and the Race to Develop Artificial Intelligence](https://www.congress.gov/crs-product/IF13051)[^21]: [DeepSeek vs ChatGPT 2026: 97.3% vs 60.3% MATH-500 and 9x ...](https://tech-insider.org/deepseek-vs-chatgpt-2026/)[^22]: [DeepSeek vs OpenAI: Which Is Better in 2026 - Interview Kickstart](https://interviewkickstart.com/blogs/articles/deepseek-vs-openai)[^23]: [Grok-3 vs DeepSeek R1 vs ChatGPT o3-mini: The AI Battle of 2026](https://www.appypieautomate.ai/blog/comparison/grok-3-vs-deepseek-r1-vs-chatgpt-o3-mini)[^24]: [Where Deepseek First Reasoning Model Beats OpenAI ... - Latenode](https://latenode.com/blog/ai/llms-models/deepseek-first-reasoning-model-beats-openai)[^25]: [Ultimate Guide - The Best OpenAI Open Source Models in 2026](https://www.siliconflow.com/articles/en/the-best-openai-models-in-2025)
---
*This report was generated automatically; citation markers `[^N]` map to
the References list above in order.*
*Full trace on Langfuse*: [https://cloud.langfuse.com/project/cmo85r1z800cnad07chmi8dt8/traces/95a1ace2abc30fc4e13531a00bb6b0b8](https://cloud.langfuse.com/project/cmo85r1z800cnad07chmi8dt8/traces/95a1ace2abc30fc4e13531a00bb6b0b8)


---

# Query 4 (EN): Compare vector databases: Qdrant vs Weaviate vs Milvus for production RAG

# Compare vector databases: Qdrant vs Weaviate vs Milvus for production RAG

> **Automated research report** — generated by Research Assistant Agent.
> *Generated at*: 2026-04-23T09:27:22+00:00 · *Cost*: $0.1705
> **Disclaimer**: AI-synthesized from public sources. Readers should verify
> before relying on this for high-stakes decisions.
## Plan overview
The original question was decomposed into 7 sub-questions:
1. **What are the key features and architecture of Qdrant vector database?**
   *Understanding Qdrant's core capabilities and design is essential for comparison.*
2. **What are the key features and architecture of Weaviate vector database?**
   *Understanding Weaviate's core capabilities and design is essential for comparison.*
3. **What are the key features and architecture of Milvus vector database?**
   *Understanding Milvus's core capabilities and design is essential for comparison.*
4. **What are the specific requirements and best practices for using vector databases in production RAG (Retrieval-Augmented Generation) systems?**
   *Identifying RAG-specific requirements helps evaluate which database features matter most.*
5. **How do Qdrant, Weaviate, and Milvus compare in terms of performance benchmarks, scalability, and query latency for RAG workloads?**
   *Performance metrics are critical for production deployment decisions.*
6. **What are the differences between Qdrant, Weaviate, and Milvus in terms of deployment options, ease of integration, and operational complexity for production environments?**
   *Operational considerations directly impact production readiness and maintenance costs.*
7. **What are the cost implications and licensing models for running Qdrant, Weaviate, and Milvus at scale in production RAG applications?**
   *Understanding total cost of ownership is essential for making informed production decisions.*

---

## 1. What are the key features and architecture of Qdrant vector database?

Qdrant is an open-source vector database written in Rust that is designed for efficient search and retrieval of high-dimensional data [^4]. It operates on collections of points as its fundamental unit of data [^1], and is optimized for high performance and scalability [^4].

The key architectural features of Qdrant include GPU-accelerated HNSW (Hierarchical Navigable Small World) indexing [^1][^2], which enables fast similarity search across high-dimensional embeddings. The database also incorporates quantization for memory optimization [^1] and supports strong metadata filtering capabilities alongside vector search [^4]. Additionally, Qdrant offers multi-tenancy support and is designed to handle operations on billions of vectors with low-latency performance [^1].

Qdrant's main capabilities include efficient similarity search, seamless integration with existing data infrastructures [^1], and support for both self-hosted and managed cloud deployments [^4]. It provides a rich API for interacting with data [^3] and is commonly used for applications such as semantic search, recommendation systems, anomaly detection, fraud detection, and retrieval-augmented generation [^1][^4]. The database is licensed under the Apache License 2.0, encouraging community contributions and transparency [^3].
## 2. What are the key features and architecture of Weaviate vector database?

Weaviate is an open-source, cloud-native vector database designed for storing, indexing, and querying high-dimensional data[^8]. It stores both objects and vectors, allowing for the combination of vector search with traditional structured filtering[^7]. The database is organized into a 3-layer architecture to achieve maximum performance, highly efficient vector search, and ACID compliance[^7].

At its core, Weaviate combines a vector indexing engine based on HNSW (Hierarchical Navigable Small World graphs) with a modular architecture[^8]. This design enables semantic search by leveraging vector embeddings—numerical representations of unstructured data like text, images, or audio[^8]. Internally, Weaviate automatically performs HNSW combined with BM25 search to deliver the closest results[^7].

Key architectural features include multi-tenancy capabilities with strong isolation and efficient resource utilization[^10]. Weaviate's multi-tenancy design channels each tenant's data through isolated pipelines down to individual properties and data types, ensuring security and efficiency[^10]. The system employs mechanisms like lazy shard loading and lazy segment loading to optimize memory usage and I/O performance during high-throughput operations[^10]. Additionally, Weaviate integrates well with frameworks like LangChain and LlamaIndex, and includes native vector search and embedding storage capabilities[^7].
## 3. What are the key features and architecture of Milvus vector database?

Milvus is an open-source distributed vector database developed by Zilliz that provides efficient storage, retrieval, and similarity search for large-scale vector data [^12][^14]. The database features approximate nearest neighbor search, real-time insertion capabilities, and horizontal scalability to handle billions of vectors [^11]. Key architectural features include a distributed deployment model that supports multi-machine scaling, cloud-native design for seamless cloud integration, and GPU acceleration using CUDA technology for enhanced performance in vector indexing and searching [^11][^14].

The architecture of Milvus 2.x (released in January 2022) represents a major redesign with a completely novel system design compared to Milvus 1.x [^15]. It incorporates coordinators and worker nodes that enable scalability, tunability, multi-tenancy, and data isolation, while an object storage layer provides data persistence [^15]. The similarity search engine relies on heavily-modified forks of third-party open-source libraries such as Faiss, DiskANN, and hnswlib [^12].

Milvus supports multiple indexing structures including graph indices (such as HNSW and CAGRA), inverted-list based indices, brute-force search, and vector quantization techniques like product quantization (PQ) and scalar quantization (SQ) [^12]. As a database, it provides column-oriented storage, four data consistency levels (including strong and eventual consistency), data sharding, multi-tenancy support, and multi-vector hybrid search capabilities [^12]. The platform offers official SDK clients for Python, Java, Go, and NodeJS, with an additional C# SDK contributed by Microsoft [^12].
## 4. What are the specific requirements and best practices for using vector databases in production RAG (Retrieval-Augmented Generation) systems?

Insufficient evidence to conclude on specific requirements and best practices for using vector databases in production RAG systems. While the evidence describes fundamental RAG processes—including text segmentation into chunks, encoding chunks into vector representations, storing them in vector databases, and performing similarity searches during retrieval [^16]—it does not provide concrete production requirements such as scalability specifications, latency benchmarks, data consistency standards, or operational best practices. The evidence mentions that Ray can speed up retrieval calls by 2x and improve scalability of RAG distributed fine-tuning [^20], but this represents only a single optimization technique rather than comprehensive production guidance. Additionally, the evidence notes that naive RAG encounters retrieval challenges with precision and recall [^16], but does not detail specific mitigation strategies or production-level requirements to address these issues.
## 5. How do Qdrant, Weaviate, and Milvus compare in terms of performance benchmarks, scalability, and query latency for RAG workloads?

**Query Latency Performance**

Qdrant demonstrates strong single-node latency performance, achieving sub-10ms query latency at a million vectors [^23][^24], with typical latencies around 10–30ms per query [^22]. Weaviate is optimized for production scale at 1M-100M vectors and achieves sub-10ms p95 latency at 100K vectors with 1536-dimensional embeddings [^25]. Milvus shows higher latencies, typically around 50ms for large-scale searches [^22], but excels in distributed, high-throughput scenarios.

**Scalability and Benchmark Range**

Qdrant leads on single-node latency in the 1M-10M vector range [^23][^24], making it ideal for smaller to mid-scale deployments that can run on minimal infrastructure like a $20/month VPS [^23][^24]. Milvus pulls ahead at 100M+ vectors with distributed deployments [^23][^24], positioning it as the choice for billion-scale workloads [^25]. Weaviate's production optimization targets the 1M-100M vector range [^25], bridging the gap between single-node and massive distributed systems.

**RAG Workload Suitability**

For RAG workloads specifically, the choice depends on scale and feature requirements. Qdrant offers low-latency performance with strong filtering capabilities [^21], while Milvus is built for large clusters and high-throughput scenarios [^21]. Weaviate distinguishes itself through native hybrid search combining vector and BM25 search [^21][^25], plus knowledge graph integration and a strong GraphQL API [^25], making it particularly developer-friendly for RAG applications that require both semantic and keyword-based retrieval.
## 6. What are the differences between Qdrant, Weaviate, and Milvus in terms of deployment options, ease of integration, and operational complexity for production environments?

All three databases—Milvus, Qdrant, and Weaviate—are open-source vector databases, which means they all support self-hosted deployment options [^26][^27]. However, Milvus offers an additional managed service alternative: Zilliz Cloud delivers the same Milvus architecture as a fully managed service with advanced features including elastic scaling, high availability, enterprise-grade security and compliance, and global deployment [^26]. This gives Milvus users a choice between open-source self-hosting and managed cloud deployment, whereas the evidence does not indicate comparable managed service offerings for Qdrant or Weaviate [^26][^27].

Regarding operational complexity, the evidence provides a clear distinction for Milvus: self-hosted Milvus deployments involve "the operational complexity of managing the database yourself," which is presented as a key consideration for production environments [^30]. The evidence suggests that using Milvus through a managed service like Zilliz Cloud or Shakudo can eliminate this operational burden [^26][^30]. However, the evidence lacks detailed comparative information on the operational complexity and integration ease of Qdrant and Weaviate for production environments, making it impossible to provide a comprehensive comparison across all three databases on these dimensions.

Insufficient evidence to conclude on specific differences in ease of integration between these three databases. While Weaviate is described as "AI-native" and built for "modern applications that combine structured and unstructured data" [^27], and the evidence mentions that Milvus users value its "performance and scalability" [^26], there are no detailed comparative assessments of integration complexity or operational requirements for Qdrant and Weaviate in production settings.
## 7. What are the cost implications and licensing models for running Qdrant, Weaviate, and Milvus at scale in production RAG applications?

*(No synthesized answer available for this sub-question.)*

---

## References
[^1]: [The Fundamentals of Qdrant: Understanding the 6 Core Concepts | Airbyte](https://airbyte.com/data-engineering-resources/fundamentals-of-qdrant)[^2]: [Understanding Embeddings and Vector Databases with Qdrant](https://www.eusoj.dev/blog/understanding-embeddings-and-vector-databases-with-qdrant)[^3]: [Unlocking the Power of Qdrant: A Comprehensive Guide to the Open-Source Vector Database - Onegen](https://www.onegen.ai/project/unlocking-the-power-of-qdrant-a-comprehensive-guide-to-the-open-source-vector-database/)[^4]: [Qdrant — Vector Database | datastores.ai](https://datastores.ai/db/qdrant)[^5]: [A Deep Dive into Qdrant, the Rust-Based Vector Database](https://medium.com/tech-ai-made-easy/a-deep-dive-into-qdrant-the-rust-based-vector-database-9f6506beabb8)[^6]: [Key Concepts & Architecture - Weaviate Academy](https://academy.weaviate.io/courses/wa050-py)[^7]: [What is Weaviate - GeeksforGeeks](https://www.geeksforgeeks.org/artificial-intelligence/what-is-weaviate/)[^8]: [High-Speed AI Indexing with Weaviate | simplyblock](https://simplyblock.io/glossary/what-is-weaviate/)[^9]: [Vector Database Architecture - Meegle](https://www.meegle.com/en_us/topics/vector-databases/vector-database-architecture)[^10]: [Rethinking Vector Search at Scale: Weaviate's Native, Efficient and Optimized Multi-Tenancy | Weaviate](https://weaviate.io/blog/weaviate-multi-tenancy-architecture-explained)[^11]: [What are the key features of Milvus?](https://linkgo.dev/faq/the-key-features-of-milvus)[^12]: [Milvus (vector database) - Wikiwand](https://www.wikiwand.com/en/articles/Milvus_(vector_database))[^13]: [Database Battle: Milvus vs. Redis Vector Benchmarks](https://myscale.com/blog/milvus-vs-redis-vector-database-benchmarks)[^14]: [An In-Depth Look at Milvus search: 10 Key Features - WPSOLR](https://www.wpsolr.com/an-in-depth-look-at-milvus-search-10-key-features)[^15]: [Introduction to Milvus Vector Database - Zilliz Learn](https://zilliz.com/learn/introduction-to-milvus-vector-database?__hstc=175614333.6694e1c5b8259356fcccdd9cfcb617fb.1767571200175.1767571200176.1767571200177.1&__hssc=175614333.1.1767571200178&__hsfp=3006156910)[^16]: [Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997v5) — 2023-12-18[^17]: [ML/AI Architect - DXC Technology - BeBee](https://bebee.com/gb/jobs/ml-ai-architect-dxc-technology-london--ss-gb-1d347t6)[^18]: [AWS First Cloud AI Journey Bootcamp 2026 - Facebook](https://www.facebook.com/groups/2262026610652241/posts/3075316832656544/)[^19]: [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](http://arxiv.org/abs/2404.16130v2) — 2024-04-24[^20]: [Retrieval Augmented Generation with Huggingface Transformers and Ray](https://huggingface.co/blog/ray-rag) — 2021-02-10[^21]: [Vector DBs, Decoded: Qdrant vs Milvus vs Weaviate | by Nikulsinh Rajput | Medium](https://medium.com/@hadiyolworld007/vector-dbs-decoded-qdrant-vs-milvus-vs-weaviate-57455146b9f6)[^22]: [Comparative Evaluation of Milvus and Qdrant for Retrieval-Augmented Generation (RAG) | by Marcus Feldman | Medium](https://medium.com/@oliversmithth852/comparative-evaluation-of-milvus-and-qdrant-for-retrieval-augmented-generation-rag-a101a72f93d1)[^23]: [Pinecone vs Weaviate vs Qdrant vs Milvus 2026 Comparison](https://www.buildmvpfast.com/blog/pinecone-vs-weaviate-vs-qdrant-vector-database-comparison-2026)[^24]: [Pinecone vs Weaviate vs Qdrant vs Milvus 2026 Comparison](https://buildmvpfast.com/blog/pinecone-vs-weaviate-vs-qdrant-vector-database-comparison-2026)[^25]: [Vector Databases for RAG: Qdrant vs Milvus vs Weaviate | Antonio Brundo](https://antoniobrundo.org/knowledge/vector-databases-rag.html)[^26]: [Milvus vs Qdrant | Vector Database Comparison - Zilliz](https://zilliz.com/comparison/milvus-vs-qdrant?__hstc=175614333.6694e1c5b8259356fcccdd9cfcb617fb.1736380800216.1736380800217.1736380800218.1&__hssc=175614333.1.1736380800219&__hsfp=38884120)[^27]: [The top 6 Vector Databases to use for AI applications in 2026 - Appwrite](https://appwrite.io/blog/post/top-6-vector-databases-2025)[^28]: [9 Most Popular Vector Databases: How to Choose](https://www.cake.ai/blog/best-vector-databases)[^29]: [We Tried and Tested 10 Best Vector Databases for RAG Pipelines](https://www.zenml.io/blog/vector-databases-for-rag)[^30]: [Top 9 Vector Databases as of March 2026 | Shakudo](https://www.shakudo.io/blog/top-9-vector-databases)
---
*This report was generated automatically; citation markers `[^N]` map to
the References list above in order.*
*Full trace on Langfuse*: [https://cloud.langfuse.com/project/cmo85r1z800cnad07chmi8dt8/traces/7ee2df72876586c89eaff9915919b466](https://cloud.langfuse.com/project/cmo85r1z800cnad07chmi8dt8/traces/7ee2df72876586c89eaff9915919b466)


---

# Query 5 (EN): What is Agentic RAG and how does it differ from traditional RAG?

# What is Agentic RAG and how does it differ from traditional RAG?

> **Automated research report** — generated by Research Assistant Agent.
> *Generated at*: 2026-04-23T09:32:01+00:00 · *Cost*: $0.1088
> **Disclaimer**: AI-synthesized from public sources. Readers should verify
> before relying on this for high-stakes decisions.
## Plan overview
The original question was decomposed into 5 sub-questions:
1. **What is RAG (Retrieval-Augmented Generation) and how does it work?**
   *Understanding traditional RAG is essential before comparing it to Agentic RAG.*
2. **What are the key components and architecture of traditional RAG systems?**
   *Knowing the components of traditional RAG helps identify what differs in Agentic RAG.*
3. **What is Agentic RAG and what are its defining characteristics?**
   *This establishes the core concept and features of Agentic RAG systems.*
4. **What are the key architectural differences between Agentic RAG and traditional RAG?**
   *This directly addresses the comparison by identifying structural and design differences.*
5. **What are the advantages and limitations of Agentic RAG compared to traditional RAG?**
   *Understanding the trade-offs helps explain why and when Agentic RAG is preferred.*

---

## 1. What is RAG (Retrieval-Augmented Generation) and how does it work?

Retrieval-Augmented Generation (RAG) is a hybrid technique in generative AI that enhances large language models (LLMs) by connecting them to external data sources [^1]. Rather than relying solely on the internal training data of an AI model, RAG systems retrieve relevant information from a knowledge base and use it to generate more accurate, context-aware responses [^1][^3]. This approach is particularly effective for tasks requiring up-to-date or specialized information [^1].

RAG works through a multi-step process. First, user queries are converted into vectors and matched against stored embeddings to fetch the most relevant data from external sources [^3]. A retriever component pulls relevant documents from a knowledge base, which can include internal databases, real-time data sources, or APIs [^2]. The LLM then uses both the original user query and the retrieved documents to generate a response that is factually accurate and context-aware [^3][^4]. Some RAG systems implement caching to check if needed data is already stored, ensuring the system consistently retrieves relevant documents and generates high-quality responses [^2].

The key advantage of RAG is that it allows LLMs to access updated knowledge and real-time information from external sources, overcoming the limitation of models trained on fixed datasets [^3]. By grounding responses in retrieved, verified external data, RAG reduces the risk of generating inaccurate information and can even provide personalized responses based on user-specific information [^3].
## 2. What are the key components and architecture of traditional RAG systems?

Traditional RAG systems are built on a fundamental architecture that connects large language models (LLMs) to external knowledge sources at query time. [^6] At their core, these systems combine a standard LLM with an external knowledge retrieval component, allowing them to retrieve relevant documents or data when a user submits a query rather than relying solely on pre-trained knowledge. [^8][^9] This approach enables RAG systems to pull in external data sources on demand and blend them with the model's generative capabilities to produce more relevant, factual, and personalized outputs. [^9]

The key architectural components of traditional RAG include a retrieval mechanism that fetches relevant information from external knowledge bases, and a generation component that uses the retrieved context to augment the LLM's response. [^6][^8] Traditional RAG systems follow a simple retrieve-and-generate pipeline for each query, [^10] where the system retrieves relevant external documents at query time to augment LLM responses. [^6] The retrieval process can employ different search strategies—modern RAG systems may use hybrid search approaches combining both dense and sparse search methods to improve retrieval quality. [^7]

A critical advantage of traditional RAG architecture is that it reduces hallucinations and improves accuracy by grounding responses in actual external data rather than relying on the model to generate information from memory alone. [^8][^9] However, the quality of a RAG system is fundamentally limited by the quality of the information it retrieves—incorrect context can mislead the LLM and result in irrelevant answers. [^8] Additionally, traditional RAG offers a cost-efficient alternative to fine-tuning because it retrieves relevant information at runtime instead of requiring new data to be added to the model itself. [^8]
## 3. What is Agentic RAG and what are its defining characteristics?

Agentic RAG (Retrieval-Augmented Generation) is an artificial intelligence approach that combines the ability to search and retrieve relevant information from vast databases with the capability to generate human-like responses based on that information.[^11] It represents an advanced version of traditional Retrieval-Augmented Generation where an AI agent retrieves external information and autonomously decides how to use that data.[^13]

The defining characteristics of Agentic RAG include autonomous decision-making and active information retrieval. Unlike traditional RAG, which performs a single retrieval followed by generation in one continuous process, Agentic RAG uses autonomous AI agents to interpret queries, plan workflows, retrieve information iteratively, refine context, and validate results.[^15] The "agentic" part specifically refers to the system's ability to act independently and decide what information to retrieve and how to use it, rather than simply pattern-matching or generating responses based on training data.[^11] Additionally, Agentic RAG can perform multi-hop retrieval and query reformulation to produce coherent, context-rich answers that would normally require manually stitching together information from several systems.[^15]

Agentic RAG systems can be implemented in different configurations, including Single-Agent RAG, which uses a single intelligent agent that routes each user query to the most appropriate data source or tool, and Tool Use Agents, which enhance standard RAG by integrating external tools like APIs or databases to fetch live or specialized data before generating responses.[^13] By optimizing the data retrieval process through these autonomous mechanisms, Agentic RAG improves the responsiveness and accuracy of AI systems for complex tasks.[^13]
## 4. What are the key architectural differences between Agentic RAG and traditional RAG?

The key architectural differences between Agentic RAG and traditional RAG center on the introduction of autonomous agency and multi-step reasoning capabilities. Traditional RAG operates as a reactive system that retrieves relevant content and generates responses based on a single user query with one-time retrieval [^17]. In contrast, Agentic RAG adds a layer of intelligence—an "agent"—on top of the retrieval process that enables the system to dynamically plan, refine, and manage multi-step retrieval and reasoning [^16]. Rather than simply gathering documents and generating answers, Agentic RAG functions as a self-learning agent capable of reasoning, planning, and acting autonomously [^18].

The functional workflow differs significantly between the two approaches. Traditional RAG follows a "Retrieve + Generate" pattern [^18], while Agentic RAG implements a "Retrieve + Reason + Act" pattern [^18]. Agentic RAG proactively plans multiple retrieval steps, reasons over gathered information, and adapts its approach dynamically to provide richer, more precise answers [^17]. Additionally, Agentic RAG performs validation, executes multi-step actions, and interacts with external tools [^20], whereas traditional RAG does not possess this capability.

The fundamental architectural distinction is that Agentic RAG endows the system with agency and autonomy [^19], making it better suited for complex, multi-step queries where tasks can be broken into parts and strategies adapted iteratively [^16]. However, this added intelligence comes at a computational cost—Agentic RAG typically uses more compute and may be slower due to iterative planning, multiple retrieval steps, and potential tool calls [^16].
## 5. What are the advantages and limitations of Agentic RAG compared to traditional RAG?

**Advantages of Agentic RAG**

Agentic RAG offers significant advantages over traditional RAG in handling complex tasks and reasoning. Unlike traditional RAG's fixed retrieve-then-generate pipeline, Agentic RAG can analyze complex queries and break them into sub-tasks, enabling multi-step reasoning [^22]. It maintains context and learns from interactions over time, rather than treating each query independently [^22]. Agentic RAG demonstrates superior adaptability—it can dynamically choose retrieval strategies (vector search, web search, API calls) based on the problem at hand [^25], and can iteratively refine its approach if initial retrieved context is insufficient [^25]. This makes it particularly suited for evolving tasks and complex, multi-step queries [^21].

**Limitations of Agentic RAG**

Agentic RAG introduces specific operational trade-offs. It typically uses more compute and may be slower due to iterative planning, multiple retrieval steps, and potential tool calls [^23]. The system faces higher costs—every "thought" step in an agentic loop consumes tokens [^23]—making it more expensive to run at scale compared to traditional RAG [^23]. Additionally, Agentic RAG introduces complexity challenges, latency concerns, data synchronization issues, and remains prone to inconsistent outputs and token limitations [^21][^22].

**Traditional RAG Advantages and Comparative Trade-offs**

Traditional RAG excels in specific scenarios where Agentic RAG's complexity is unnecessary. It offers simplicity, speed, and lower initial investment and maintenance costs [^21]. Traditional RAG delivers fast response times for simple queries and reliable performance for static information retrieval [^21], making it cost-effective for fixed tasks and straightforward knowledge bases [^23]. The key trade-off is clear: Agentic RAG sacrifices speed and cost efficiency to gain accuracy, adaptability, and reasoning depth for complex tasks [^23][^24], while traditional RAG prioritizes operational efficiency and cost-effectiveness for routine, static tasks [^21].

---

## References
[^1]: [What is retrieval-augmented generation (RAG)? · GitHub](https://github.com/resources/articles/software-development-with-retrieval-augmentation-generation-rag)[^2]: [A guide on how retrieval-augmented generation (RAG) works](https://www.merge.dev/blog/how-rag-works)[^3]: [What is Retrieval-Augmented Generation (RAG) ? - GeeksforGeeks](https://www.geeksforgeeks.org/nlp/what-is-retrieval-augmented-generation-rag/)[^4]: [Retrieval augmented generation (RAG) and indexes - Microsoft Learn](https://learn.microsoft.com/en-us/azure/foundry/concepts/retrieval-augmented-generation)[^5]: [ELI5 What is a is Retrieval-Augmented Generation ...](https://www.reddit.com/r/explainlikeimfive/comments/1p39v3g/eli5_what_is_a_is_retrievalaugmented_generation/)[^6]: [RAG Architecture From Naive Pipelines to Agentic | Galileo](https://galileo.ai/blog/rag-architecture)[^7]: [Navigating RAG System Architecture: Trade-offs and Best Practices for Scalable, Reliable AI Applications - DEV Community](https://dev.to/satyam_chourasiya_99ea2e4/navigating-rag-system-architecture-trade-offs-and-best-practices-for-scalable-reliable-ai-3ppm)[^8]: [RAG LLM Architecture: Transforming AI with Dynamic Knowledge Integration | SaM Solutions](https://sam-solutions.com/blog/rag-llm-architecture/)[^9]: [RAG Architecture Explained: A Comprehensive Guide [2026]](https://orq.ai/blog/rag-architecture)[^10]: [Agentic RAG vs Traditional RAG: Complete Guide - Mem0](https://mem0.ai/blog/agentic-rag-vs-traditional-rag-guide)[^11]: [What Is Agentic RAG? Learn About Retrieval-Augmented Generation in AI | Coursera](https://www.coursera.org/articles/agentic-rag)[^12]: [What Is Agentic RAG? | Salesforce UK](https://www.salesforce.com/uk/agentforce/agentic-rag/)[^13]: [What is Agentic RAG? - GeeksforGeeks](https://www.geeksforgeeks.org/artificial-intelligence/what-is-agentic-rag/)[^14]: [What is Agentic RAG? Building Multi-Agent Agentic RAG Systems](https://medium.com/@adeniyi221/what-is-agentic-rag-building-multi-agent-agentic-rag-systems-88ba5fa3eaf4)[^15]: [Agentic RAG : A comprehensive guide](https://www.kore.ai/blog/what-is-agentic-rag)[^16]: [Traditional RAG vs Agentic RAG: Key Differences](https://www.xcubelabs.com/blog/traditional-rag-vs-agentic-rag-key-differences/)[^17]: [Agentic RAG vs Traditional RAG: Key AI Differences - Softude](https://www.softude.com/blog/agentic-rag-vs-traditional-rag/)[^18]: [Traditional RAG vs. Agentic RAG: The Next Evolution of AI](https://hyqoo.com/artificial-intelligence/traditional-rag-vs-agentic-rag)[^19]: [Agentic vs. Traditional Retrieval-Augmented Generation - Medium](https://medium.com/@adnanmasood/beyond-retrieval-agentic-vs-traditional-retrieval-augmented-generation-9ee50c8242c2)[^20]: [Traditional RAG vs Agentic RAG - What’s the Difference?  - DataMites Offical Blog](https://datamites.com/blog/traditional-rag-vs-agentic-rag-whats-the-difference/?srsltid=AfmBOorlWPHV5UKJuYNTrfeArxBW4XLJLoczz-FxJPSrJRCvmiD9xvPp)[^21]: [Traditional RAG and Agentic RAG Key Differences Explained - TiDB](https://www.pingcap.com/article/agentic-rag-vs-traditional-rag-key-differences-benefits/)[^22]: [Agentic RAG vs Traditional RAG: Complete Guide - Mem0](https://mem0.ai/blog/agentic-rag-vs-traditional-rag-guide)[^23]: [Traditional RAG vs Agentic RAG: Key Differences - [x]cube LABS](https://www.xcubelabs.com/blog/traditional-rag-vs-agentic-rag-key-differences/)[^24]: [Agentic RAG : A comprehensive guide](https://www.kore.ai/blog/what-is-agentic-rag)[^25]: [Traditional vs Agentic RAG: Limitations and Benefits | Akshay Pachaar posted on the topic | LinkedIn](https://www.linkedin.com/posts/akshay-pachaar_traditional-vs-agentic-rag-clearly-explained-activity-7361738628479270912-86mR)
---
*This report was generated automatically; citation markers `[^N]` map to
the References list above in order.*
*Full trace on Langfuse*: [https://cloud.langfuse.com/project/cmo85r1z800cnad07chmi8dt8/traces/579fb076779afed23d3f620fcaa20a53](https://cloud.langfuse.com/project/cmo85r1z800cnad07chmi8dt8/traces/579fb076779afed23d3f620fcaa20a53)
