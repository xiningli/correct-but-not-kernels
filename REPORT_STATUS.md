# Report status

<!-- BEGIN PAPER DRAFT 20261025 -->
## 论文初稿、训练轨迹分析与公开数据包 — 2026-10-25 UTC

只投 Kaggle Gemma 4 Paper Track（截止 2026-11-12 23:59 UTC）。官方页面核验：3,000 词上限、
五个必需部分、五项等权评分；**Paper Track 无模型要求**（QAT Gemma 4 要求属主赛）。
快照与说明：[evidence/paper-rules-20261025/](evidence/paper-rules-20261025/NOTES.md)。
未租机；全部为本机 CPU 分析加一次本机 GPU 容器探针。新代码在 `research/paper_v1/`
（不放入 `eval/`，以免改动 R3 冻结源码清单；见下方遗留问题）。

**1. 收益分解**（v4 同程序复评 5,600 条，profiler 归因；任务 bootstrap 95% 未校正；不改原评分）：

| 正确且… | C − base | SFT − base | C − SFT |
|---|---:|---:|---:|
| 无声明 kernel（纯 PyTorch） | +5.54 [2.83, 8.25] | +9.83 [6.46, 13.29] | −4.29 [−6.83, −1.83] |
| 声明但未启动（装饰 kernel） | +11.62 [8.38, 14.96] | +6.83 [3.46, 10.25] | +4.79 [2.21, 7.33] |
| 实际启动 | −0.04 [−0.75, 0.58] | −0.29 [−1.17, 0.42] | +0.25 [−0.33, 0.92] |

实际启动且正确的 62 个程序全部仍用 ATen 做 conv/GEMM，自写 kernel 只做逐元素/归约尾段
（中位数约 1/3 次 launch），属部分融合而非 dummy。

**2. 训练轨迹**（R3 归档中 6 条 GRPO rollout、3 份 SFT 数据、960 个 base 候选；静态分类）。
静态分类先对 5,600 条运行时轨迹验证：“仅定义处出现” 1,519/1,519 均未启动；“可达” 90/102 启动。
LayerNorm 上 base：装饰 58%、通过率 80%；真正调用自写 kernel 4%、通过 2/39。
GRPO-C 第 1–40 步装饰占 73%；SFT 训练集 81% 为装饰。**初稿中“RFT 数据偏纯 PyTorch”的推测
被数据否定，已删除**。一个种子的 320 个 base 样本中，纯 PyTorch 崩溃 63 次，其中 58 次因调用
不存在的 Triton 属性（如 `triton.compiler_hints`）。

**3. 31B tanh 修复 GPU 复核**：同 portability-v1 隔离容器、本机 RTX 5070 Ti 执行原候选；
|x| ≤ 88 正确，|x| ≥ 89 在 fp32/fp16/bf16 下每行均非有限值。开发证据，不回写原成功评分。
[receipt](evidence/paper-probes-20261025/tanh-boundary.json)

**4. 公开数据包**：`research/paper_v1/release.py` 生成 `runs/release-paper-v1-20261025/`
（376 文件，548,082,743 bytes，含 MANIFEST/SHA256SUMS/数据卡）。
`research/paper_v1/reproduce.py` 仅凭数据包重建 R3 主报告与 4.2–4.3 全部分析，
三项与已提交回执完全一致（PASS）。当前包由未提交工作区生成（manifest 标记 dirty），
提交后应重建。权重、adapter、KernelBench 源码不在包内。

**5. 论文**：[paper/writeup.md](paper/writeup.md)，2,829 词（含参考文献）；
图 `paper/figure-decomposition.png` 由 `research/paper_v1/paper_figure.py` 从回执生成。
参考文献经联网核对，并补入 Kevin 已报告的 PyTorch fallback、robust-kbench、KernelBench-Verified；
新颖性改为“量化 + 训练动态 + RL≈RFT”，不声称首次发现 fallback。

**遗留问题（未修改）**：`tools.replication.source_hashes()` 封存 `eval/` 等目录下全部文件，
此前已提交的 `eval/portability_audit.py`、`scripts/repair_v8_autodl.sh`、
`scripts/repair_v8_pilot_autodl.sh` 使 `verify_frozen('runs/replication-v1.1')` 在 HEAD 失败，
也导致 `research/diagnostics_resume_v6/test_resume.py` 一项测试失败（冻结文件本身无改动，仅新增）。
`full_audit_20261018 final-report` 因此无法按原入口运行；`reproduce.py` 绕过源码清单检查直接重建报告。
CPU 测试：除该项外 224 通过（含新增 `research/paper_v1/test_paper.py` 5 项）。

**数字核对**：`python -m research.paper_v1.claims` 从回执重算论文中每个定量表述并要求逐字出现，
现 0 缺失；9 项引自早期报告的数字列出来源供人工核对。由此更正一处：原"960 个 base 程序中
466 个提到 Triton"取自 R3 held-out 统计却放在 LayerNorm 语境，改为 LayerNorm 613 定义/39 调用、
Level-2 274 声明/12 launch。已确认 base 候选与 GRPO rollout 的 prompt、采样参数、单次检查一致。
文风与审稿风险清单见 `paper/REVIEW-NOTES.md`。
<!-- END PAPER DRAFT 20261025 -->

<!-- BEGIN PORTABILITY V1 -->
## 消费级显卡跨设备复评：已完成并核验 — 2026-10-21 UTC

本机 RTX 5070 Ti 复评于 2026-10-20T13:36:32.177336+00:00 完成，用时 5.13 小时。
原 100 道 Level-2 任务中，78 道通过预先冻结的参考资格检查；
Base + 三个 C/RL 种子 + 三个 SFT 种子，每题八个原始样本，共 **4,368 次**逻辑尝试，全部题组完整。
未训练、未重新生成程序；比较对象是保存的同一批源码与输入种子。

| 策略 | 本机数值通过率 | 远程同任务通过率 | 本机通过 / 全部尝试 | 数值 unknown |
|---|---:|---:|---:|---:|
| Base | 54.01% | 54.01% | 337 / 624 | 223 |
| C/RL（三种子） | 71.69% | 71.63% | 1342 / 1872 | 355 |
| SFT（三种子） | 71.47% | 71.42% | 1338 / 1872 | 364 |

通过率按所有尝试计分，格式失败、崩溃及数值 unknown 保留在分母；unknown 不解释为已观察到数值错误。
远程列使用 v4 相同程序、相同任务子集，不能与原始全 100 题 v1.1 百分比直接混比。

| 本机配对比较 | 提升（百分点） | 98.333% 任务聚类 bootstrap 区间 |
|---|---:|---:|
| C-base | +17.68 | [+12.39, +23.13] |
| SFT-base | +17.47 | [+12.39, +22.49] |
| C-SFT | +0.21 | [-2.88, +3.47] |

收益在本机合格子集上保留；C/RL 与 SFT 的区间跨零，尚无 RL 优势证据，也不据此宣称等效。
三种子在题内平均，共享 Base 只计一次；区间条件于这些模型和任务，不增加独立训练样本数。

**严格自定义 kernel 成功仍为 0 / 4,368。** 原 prompt 允许 PyTorch；以上是数值正确性收益，
尚不能解释为自定义 CUDA/Triton 能力增强，也未测量加速。两张 GPU 同为 Blackwell CC 12.0，
这里只支持同架构平台复评，不支持跨架构或新任务泛化、纯硬件因果效应。

22 道排除题全部触及预设存储限制：6 道累计参考输入/输出上限、13 道 worker 磁盘上限、3 道单输出上限。
所有策略统一排除。这不证明这些题无法在 16GB 卡上运行，不能笼统写成显存不足。
数值三态（通过/失败/unknown）有 8 个跨平台不一致：1 个通过变数值失败、3 个远程受阻变本机通过、
4 个远程数值失败变本机非法显存访问崩溃。均保留原结果；日志不能单独建立迁移机制，尚未做额外 GPU 最小复现。

核验通过：5,438 个冻结输入文件、全部源码/任务/种子绑定、完整计分网格、报告与 bootstrap 重算、终态归档 SHA256。
归档：`runs/portability-v1-20261020/run-r3/evidence.tgz`，40,837,846 bytes，
SHA256 `eaf9f02441458cda14e73d8879f1fc4b80a178bff3435dfdc213ffa4299eb84e`。
本次现场检查无实验进程或容器，GPU 利用率 0%、显存约 15 MiB；本机保持开机。

[完成报告与核验](evidence/portability-v1-complete-20261021/SUMMARY.md) ·
[逐项统计与差异记录](evidence/portability-v1-complete-20261021/audit.json) ·
[复算脚本](eval/portability_audit.py) ·
[原冻结方案](research/portability_v1/README.md) ·
[启动与失败修订证据](evidence/portability-v1-launch-20261020/)
<!-- END PORTABILITY V1 -->

<!-- BEGIN V8 MEMCHECK CALIBRATION -->
## r6 前置验证中止；误判已修订，待远程 GPU 复验 — 2026-10-19 19:22 UTC

**当前没有在运行的模型实验；本轮 Gemma 输出为 0。** r6 于 18:46 UTC 写出
blocked 结果，在本地备份确认后进入关机流程。18:48 UTC 监控记录连续连接失败；
本次实时 SSH 检查仍超时。未通过 AutoDL 控制台核验平台电源或计费状态。

阻塞发生在 AttnRes 官方修复正对照，而非模型补丁：可见用例 2/2、隐藏用例
10/10、上游回归 27/27 通过；memcheck 的 10 个用例数值及输入保留检查全部
通过，工具也全部报告零错误。但它们被评测器统一判为
`undeclared_host_tensor_operation`，故 r6 原始综合结果仍为失败，未改写。
GroupNorm、RWKV7 在本轮新执行器的完整对照尚未执行。

已确认误判原因：profile 模式校准的操作列表被错误用于独立 memcheck 模式；
后者对同一官方补丁观察到了 `aten.copy_.default` 和
`aten.new_empty_strided.default`，前者列表未包含它们。
本地 r7 按两种模式分别从可信对照校准，并在任何模型输出前冻结契约；候选不能
扩充列表，缺少对应模式的契约会直接中止。所有组获得相同新增公开契约字段。
数值、输入保留、kernel 路径、回归与显存检查要求仍保留。

本地 **36 项测试通过**；新部署包已经逐文件读回验证，尚未上传或在 GPU 运行。
任务快照、测试输入、模型、随机种子及生成预算未变；历史准备清单与失败日志保留。
r7 包：`runs/repair-v8-pilot-20261019/autodl-v8-pilot-r7.tgz`，20,862,688 bytes；
SHA256 `33e4ac4cf06bee91a5063ac0b9276f1c089f69d25fec317db9222c62520a3593`。
下次有卡开机后需重新跑三题完整前置验证，通过后才能开始 24 个模型修复 episode。
目前不能声称整个环境已修好，也没有新增模型能力结论。

r6 证据归档 241,522 bytes，SHA256
`f9ac89e2246cea3be2e7e077232ad5d1bab8dafb0ecb694720dd5b6409a74062`；
已重新核验清单中全部 230 个文件，历史成果未被本次失败覆盖。

[阻塞诊断](evidence/repair-v8-memcheck-calibration-20261019/diagnosis.json) ·
[显存检查原始结果](evidence/repair-v8-memcheck-calibration-20261019/fixed-memcheck.json) ·
[备份校验](evidence/repair-v8-memcheck-calibration-20261019/backup-audit.json) ·
[修订声明](evidence/repair-v8-memcheck-calibration-20261019/amendment.json) ·
[部署包回执](evidence/repair-v8-memcheck-calibration-20261019/bundle-receipt.json)
<!-- END V8 MEMCHECK CALIBRATION -->

<!-- BEGIN V8 RUNTIME REPAIR -->
## 执行器修复后重新验证，模型尚未生成 — 2026-10-19 18:30 UTC

重新开机后已通过系统调用追踪定位并修复启动阻塞，当前部署为
`/root/autodl-tmp/repair-v8-pilot-20261019-r6`。所有修订均发生在模型生成前，
三题代码快照、输入、参考计算、提示词、随机种子及生成预算保持一致；之前的失败
版本和日志单独保留。新准备清单 SHA256 为
`4b3bf62cb3b9b47d22ee3758bcc90be1b3087f5839d928018d8393e5fdbd55a4`。

修复内容：

- 仅允许 LLVM/PyTorch 所需的 `prctl(PR_SET_NAME/PR_GET_NAME)` 线程名操作；
  其他 prctl 及候选网络调用仍被拒绝。可信 requests 依赖在 seccomp 前完成
  urllib3 的 IPv6 导入探测，不为候选代码开放网络。
- Compute Sanitizer 必须包裹真正执行 CUDA 的 Python 进程；其子进程内在
  导入 torch/候选前绑定 Landlock，避免 `/proc/self` 规则错绑到工具父进程。
  可信工具本身位于 Landlock 外，仍受专用 UID/资源限制，属于评测可信依赖。
- Kineto 与 Compute Sanitizer 不能同时订阅 CUPTI，改为对同一个已封存补丁、
  相同输入面板分别做 kernel/操作观察和 memcheck。正式成功判定仍要求可见、隐藏
  profiling、上游回归、memcheck 全部通过；内存检查单独通过不能判修复成功。
- 必须得到明确的零错误内存检查摘要；增加真实越界写反例、GPU 恢复及在
  Sanitizer 下读取隐藏参考被拒绝的检查，防止“工具启动了但没检测”的假通过。

已完成 22 项 pilot CPU 测试（加原有资格/监督 8 项共 30 项），34 个 CPU 参考用例
检查通过。远程六项普通隔离检查通过；Sanitizer 正常 kernel 为零错误，故意越界写
返回 86 并报告 514 个错误，随后 GPU 恢复成功；其隐藏参考读取检查也通过。
AttnRes 官方修复的单个可见缺陷用例已在新执行器通过。完整三题 before/fixed、隐藏、
回归与 memcheck 正在运行，**尚未得出任何新模型能力结论**。

后台 PID 2815 执行本轮完整对照验证，PID 3134 等待验证结果；通过后自动进入完整
实验控制器。失败时保留 10 分钟现场诊断窗口，若未接管则归档并关机；完整实验完成
或阻塞也会生成报告、等待本地校验备份、关机。当前本地监控为
`runs/repair-v8-pilot-20261019/remote-r6/status.json`。
已先行备份 229 个诊断/验证文件，归档 **3,350,188 bytes**，SHA256
`3b214c99c2b122b9a06fdf4f505068fbeb08b043beb3c489f9e7f4e8551379c9`。

[修订与输入不变声明](evidence/repair-v8-runtime-repair-20261019/amendment.json) ·
[隔离验证](evidence/repair-v8-runtime-repair-20261019/isolation-controls.json) ·
[显存检测正反例](evidence/repair-v8-runtime-repair-20261019/sanitizer-controls.json) ·
[本地备份](evidence/repair-v8-runtime-repair-20261019/backup-audit.json)
<!-- END V8 RUNTIME REPAIR -->

<!-- BEGIN V8 THREE TASK PILOT OPERATION -->
## 三题修复实验启动验证阻塞，证据已备份并请求关机 — 2026-10-19 18:07 UTC

启动包已上传、核验 SHA256 并部署到
`/root/autodl-tmp/repair-v8-pilot-20261019`。后台控制器 PID 为 1753，运行标识为
`20261019T180544.370170Z`。启动时 GPU 无其他计算任务；RTX PRO 6000 96GB、
既有评测与推理环境均已通过实际 CUDA 运算检查。

CPU 环境、62.6 GB 模型权重与 tokenizer 一致性预检全部通过。但首个可信 Triton
smoke 在新隔离执行器中被 `SIGSYS` 终止（returncode -31），尚未得到
`CUDA_TRITON_OK`。该 smoke 用时 2.72 秒；整个启动流程约 95 秒。
**Gemma 生成数为 0；这是评测基础设施阻塞，不是模型修复失败。**
尚未进入三题 before/fixed、回归、memcheck 或模型对照实验，也没有新增能力结论。

本地独立监控已下载 **177,662 bytes** 证据归档，核验整体 SHA256 和清单中的
**45 个文件**，随后发送备份确认。控制器按“阻塞后关机”的既有授权进入关机流程，
此后 SSH 会话中断且连续三次连接失败；未通过 AutoDL 控制台验证平台电源/计费状态。
归档 SHA256：`1625367728ce787409883f8b85a236cfd9347c54c16add428c88b999f97bb934`。
远程数据盘保留原始证据，本地副本位于
`runs/repair-v8-pilot-20261019/remote/20261019T180544.370170Z/evidence.tgz`。

初次日志没有记录被拦截的具体系统调用，因此尚不能声称问题已经定位或修复。
本机 Torch 2.11 / Triton 3.6 使用相同 seccomp 规则的简化 smoke 通过，但版本不同、
未复现远程 UID/Landlock，不能替代远程验证。已准备独立的
`scripts/repair_v8_pilot_diagnose.py`：下次有卡开机后，在原规则下用 strace 运行
可信 smoke，记录 `si_syscall`，再据此修复；不放宽过滤、不产生模型输出、不写 PASS。
首次失败包与证据保留，修订后的执行器必须重新准备 manifest 并通过全部关卡。

[部署回执](evidence/repair-v8-pilot-launch-20261019/deployment-receipt.json) ·
[环境及模型预检](evidence/repair-v8-pilot-launch-20261019/runtime.json) ·
[阻塞日志](evidence/repair-v8-pilot-launch-20261019/controls.log) ·
[备份核验](evidence/repair-v8-pilot-launch-20261019/backup-audit.json) ·
[下一步诊断](evidence/repair-v8-pilot-launch-20261019/next-step.json)
<!-- END V8 THREE TASK PILOT OPERATION -->

<!-- BEGIN V8 THREE TASK PILOT PREPARATION -->
## 三题 Gemma 修复实验代码与启动包准备完成 — 2026-10-19 UTC

**现在可以为下一批开发实验插卡开机。** 本地已完成代码、输入包、CPU 检查和
独立目录解包验证；本轮未连接 AutoDL、未执行 GPU 任务，也没有新增模型结果。
开卡后先验证新评测器，所有关卡通过才进入 Gemma 生成。

本批仅使用已合格的 AttnRes、GroupNorm、RWKV7 三个开发缺陷，覆盖两个仓库。
同一个 Gemma-4-31B-it BF16、thinking=false，比较四组：完整模块/局部结构化编辑
分别搭配简略反馈/结构化诊断。每组每题 2 个采样副本、最多 4 次修改，共 24 个
逻辑 episode、最多 96 次逻辑尝试；首轮 A/B、C/D 分别共享抽样和评分，最多
84 次物理生成，每个逻辑 episode 仍承担完整首轮 token/执行费用。
重启原代码的 E 组留到后续十二题开发阶段；本批不训练，也不代替正式独立测试。

已实现：

- 两种格式共用同一修改边界，支持函数/方法、内部签名与调用点、import、已有顶层
  常量及 `_repair_` helper；保护公共签名、类声明、许可证和未列出的文件。
- 每轮最多 16,384 个输入 token、8,192 个输出 token。拒绝格式、越界修改、过期
  源码 hash、未正常停止的输出都会消耗尝试。反馈只投影可见执行字段。
- CPU 端独立计算输出与梯度参考；候选只获得原仓库快照、补丁及当前输入。
  检查原仓库调用、输入保留、输出与梯度、边界/guard、原仓库回归和 memcheck。
  本批明确要求原目标 kernel 符号仍在执行路径；允许的宿主 ATen 操作在模型生成前
  由可信对照封存并公开给所有组。它是有限修改范围的开发实验，不覆盖任意重写。
- 保留 FLA 原有 `torch.compile` helper，不允许借此新增编译包装器绕过观察。
  启动门控含六项隔离检查、官方补丁正对照、高层替代及 dummy kernel 负对照。
- 逐轮串行切换推理环境与评测环境，避免同时占用 GPU；保存请求、响应、token、
  时间、代码及失败签名。JSON 决策使用原子写入；恢复时保留已保存生成，重试执行
  只能使用同一补丁和输入，保留中断记录。
- 所有 episode 的补丁先按可见反馈选定并封存，再开始隐藏评分；不从隐藏成绩选
  best-of-n。报告分别列出可靠修复、可见通过但最终失败、观察到的语义回归和错误
  签名重复，并区分逻辑成本和实际生成工作。三题不计算正式显著性结论。
- AutoDL 专用控制器限制总运行 12 小时，完成或运行中阻塞后写报告、核验归档，
  等待本地备份确认，随后执行既有授权的关机；数据盘保留原始证据。

本地验证结果：25 项测试通过；34 组 CPU 输入/参考计算通过；三个官方修复均可由
两种编辑接口表达。完整原模块 JSON 输出分别为 4,712、3,949、4,183 tokens，低于
6,144 的准备上限；初始提示词约 4,845–6,243 tokens，启动时还会检查加入运行契约后的
实际提示词。独立解包目录中重新核验全部 hash，17 项 pilot 测试再次通过，Shell
入口语法检查通过。**新隔离环境、扩展 GPU 用例和推理环境的实际检查仍待开卡。**

启动包 `runs/repair-v8-pilot-20261019/autodl-v8-pilot.tgz` 为 **20,859,635 bytes**，
含 2,392 个文件/软链接，已逐文件读回核验。SHA256：
`05e5281c7e59b84915d5950425a8ef4bdda008b46b1c1d7cf8b589647264a8e5`。
这是研究者部署包，包含官方修复对照，不能整体暴露给模型或候选执行进程。

计划部署到 `/root/autodl-tmp/repair-v8-pilot-20261019`，复用既有 31B 模型、旧推理
环境和单独锁定的资格验证环境。入口为
`bash scripts/repair_v8_pilot_autodl.sh run`；CPU 环境检查可单独使用 `prepare`。
这次没有部署到远程机器，也没有启动后台任务。

[CPU 准备回执](evidence/repair-v8-pilot-preparation-20261019/cpu-preflight.json) ·
[源码与输入清单](evidence/repair-v8-pilot-preparation-20261019/manifest.json) ·
[启动包回执](evidence/repair-v8-pilot-preparation-20261019/bundle-receipt.json) ·
[独立解包验证](evidence/repair-v8-pilot-preparation-20261019/portable-check.json) ·
[部署计划](evidence/repair-v8-pilot-preparation-20261019/deployment-plan.json)

后续八个开发候选、暂缓的 FlagGems 和正式 60–80 题研究仍未完成；不能将本批
“代码准备完成”解释为全部研究准备完成。历史 v1–v7 和首批资格验证证据保持不变。
<!-- END V8 THREE TASK PILOT PREPARATION -->

<!-- BEGIN V8 GPU BATCH1 OPERATION -->
## 首批 GPU 资格验证完成：3 题合格，1 题环境阻塞 — 2026-10-19 UTC

**本批 4 个开发候选全部执行完毕：Liger AttnRes、Liger GroupNorm、FLA RWKV7
通过资格检查；FlagGems addmm 在当前锁定环境下存在运行接口错误，暂不合格。**
正式封存测试题仍为 0，本轮 Gemma 运行和训练均为 0；这些结果验证的是题目与评测
入口的可靠性，尚不能增加任何模型能力收益结论。合格题覆盖两个仓库，仍缺第三仓库。

GPU 阶段从 01:20:22 到 01:41:44 UTC，约 21 分钟。执行了计划中的全部 **204 个
独立进程**：24 次原仓库测试、156 次独立探针、12 次 kernel trace、12 次 memcheck。
没有遗漏、重复、自动重试或超时，试验过程中未改动已冻结的用例、容差或上游代码。

| 开发题 | before 主触发（3 次） | fixed 独立用例 | 原仓库回归（两版本合计） | fixed memcheck | 资格 |
|---|---|---|---|---|---|
| Liger #1344 / AttnRes | 3/3 数值不符 | 18/18 通过 | 162/162 通过 | 3/3 通过 | PASS |
| Liger #1376 / GroupNorm | 3/3 反传 shape 错误 | 18/18 通过 | 30/30 通过 | 3/3 通过 | PASS |
| FLA #1103 / RWKV7 | 3/3 guard 被越界写入 | 18/18 通过 | 24/24 通过 | 3/3 通过 | PASS |
| FlagGems #553 / addmm | 未获得目标缺陷复现 | 24 次均遇运行接口错误 | 270 次失败 | 程序未通过 | HOLD |

计数是同一开发题不同用例和重复进程，**不构成数百个独立缺陷**。三个合格题的
before 保留行为检查也全部通过，固定三个 seed 为 1701、1702、1703；每题的三次
profiler 都确认了原自定义 GPU kernel 实际执行。FLA 非对齐完整路径还出现数值偏差
或非有限输出，主资格触发采用预先固定的 guard 检查，未根据这些事后表现换终点。

### FlagGems 暂缓原因

before/fixed 的 48 次独立探针均出现
`TypeError: function takes exactly 16 arguments (15 given)`。追踪经过原包的
`utils/libentry.py` 与 Triton launch 路径；addmm autotune 清缓存触发注册后的
`fill_scalar_` 路径时即报错。原有 addmm/mm 回归也在同一环境失败。
这是当前原仓库运行栈的阻塞，不能用它证明 bias 修复无效，更不能算模型修复失败。

虽然三个 FlagGems sanitizer 日志也打印了 `0 errors`，程序均以非零状态退出且
没有成功完成目标计算，资格门控仍全部拒绝通过。没有通过摘出 kernel、关闭原库
集成检查、修改生产包装层或替换人类补丁来绕过问题。该题保留为开发暂缓项；后续
若更换兼容依赖或硬件，须单独记录环境修订和新结果，不能覆盖本次失败证据。

### 环境、证据与关机

AutoDL 是 RTX PRO 6000 Blackwell Server Edition（97,887 MiB），驱动 580.82.09；
独立环境为 Python 3.12.3、PyTorch 2.8.0+cu128、Triton 3.4.0，使用 CUDA 13.0
工具集提供的 compute-sanitizer。数据盘部署时剩余约 301 GiB。原实验与模型目录
保持不变，新目录是 `/root/autodl-tmp/repair-v8-batch1-20261019T011025Z`，运行 ID
`20261019T012023.952866Z`。部署前已核对源码包 5,948 个文件/软链接，远程 CPU
参考计算 26/26、原包导入/测试收集 8/8 通过。

[完整运行证据包](evidence/repair-v8-gpu-qualification-batch1-20261019/evidence.tgz)
为 867,849 bytes，SHA256
`1144d82ec7e368b296dc2408483b77193f93462adb1d3d4e18eb5d97c7eefbc8`。
已拉回本地：包内 544 个文件解包通过，204 个进程记录与预定矩阵一一对应，204 份
日志哈希和 JSON 回执核对通过。冻结代码/依赖锁、执行源码清单均与本地准备一致，
四个资格判定已从原始记录独立重算，结果一致。数据盘也保留原始证据。

本地报告和备份核验完成后，已上传 SHA256 备份确认，触发用户授权的自动关机。
随后原 SSH 会话关闭，两次新连接检查均未取得 SSH banner。未重启主机；关机返回
回执未能在连接断开前取回，未查询平台电源/计费状态。见
[关机观察记录](evidence/repair-v8-gpu-qualification-batch1-20261019/shutdown-observation.json)。

[任务判定](evidence/repair-v8-gpu-qualification-batch1-20261019/task-summary.json) ·
[备份与独立审计](evidence/repair-v8-gpu-qualification-batch1-20261019/backup-audit.json) ·
[失败类型记录](evidence/repair-v8-gpu-qualification-batch1-20261019/observed-probe-failures.json) ·
[环境回执](evidence/repair-v8-gpu-qualification-batch1-20261019/environment.json)

下一步在本地完成三个合格开发题的可见/隐藏测试划分与公平编辑接口，并继续准备
余下 8 个候选，优先补齐第三仓库覆盖。当前尚未启动正式模型对照；不需要为这些
CPU 准备工作持续保留运行中的 GPU。
<!-- END V8 GPU BATCH1 OPERATION -->

## 首批 4 题本地准备完成，待 GPU 资格验证 — 2026-10-18 UTC

**现在可以为首批 4 个开发候选启动 AutoDL GPU。** 本地已准备完整的原仓库
before/fixed 快照、独立环境、复现与回归入口及可移植启动包。这里的“准备完成”
只覆盖首批 4 题；12 个开发候选中的其余 8 题尚未制作运行包。
**GPU 已运行次数、GPU 合格题数、正式封存题数、Gemma 运行次数均为 0。**
本轮没有连接或启动远程主机。

首批从已有开发队列选出，依据是三个仓库的覆盖、有限 API 和可用原仓库测试，
没有使用模型表现或 GPU 结果选择题目。它们继续属于 development，不能转为正式未见测试题。

| 开发候选 | 缺陷 | 独立用例 | 原仓库测试收集数（每个版本） |
|---|---|---:|---:|
| Liger #1344 | AttnRes 超过 32 个 block | 6 | 27 |
| Liger #1376 | GroupNorm 多维反传分配 | 6 | 5 |
| FlagGems #553 | addmm 矩阵 bias 广播 | 8 | 45 |
| FLA #1103 | RWKV7 尾部越界 | 6 | 4 |

表中最后一列仅是 CPU 上的测试收集数，**不是 GPU 测试通过数**。before/fixed
使用相同的 before 测试树做保留行为检查，生产代码来自各自真实提交；外部独立用例
覆盖新触发条件。保留完整包导入与公开调用路径，没有摘取 kernel、补 stub 或修改
上游生产代码。相比上一节仅静态筛选，本轮已经实际 import 上游包并执行测试收集。

本地验证结果：

- 8 份完整源码快照与已筛选归档逐文件核对通过，源码和执行快照均有 SHA256 清单。
- 独立 Python 3.12.13 环境安装了 53 个锁定依赖；依赖兼容检查通过。
  采用 PyTorch 2.8.0+cu128、Triton 3.4.0、Transformers 4.57.1；PyTorch 官方列有
  [2.8.0 / CUDA 12.8 的 Linux 安装组合](https://pytorch.org/get-started/previous-versions/)。
  这是任务验证环境，不修改已有训练环境，也不把主机 CUDA Toolkit 版本当作 wheel runtime。
- 8/8 原包导入通过；8/8 测试收集通过，前后版本测试数匹配。
- 26/26 CPU 参考计算用例通过；7 个 CPU 单元/负控制通过，覆盖独立梯度公式、遗漏
  block、越界 guard、缺少重复实验、CPU 冒充 GPU、回归失败和不安全归档路径。
- 无 CUDA 时 GPU 入口明确拒绝启动，未创建 GPU 运行目录。
- 启动包全部 5,948 个文件/软链接读回验证通过；换目录解包后重建 8 份执行快照，
  计划语义及代码哈希一致，7 个 CPU 控制再次通过。尚未在 AutoDL 上安装或运行此环境。

GPU 资格入口固定三个 seed，每次新起进程，串行使用一张 GPU。每题必须同时满足：
before 三次稳定触发预定缺陷；fixed 三次通过全部独立用例；两版本通过原仓库选定
回归测试和保留行为检查；profiler 确认原自定义 kernel 实际执行；compute-sanitizer
memcheck 通过。导入失败、缺卡、OOM、超时、空测试和全部 skip 不能记为缺陷复现或
资格通过。原仓库 flaky 自动重跑插件被关闭，保留失败记录。

研究人员启动包：
[`autodl-batch1-investigator.tgz`](runs/repair-v8-qualification-20261018/autodl-batch1-investigator.tgz)，
39,794,876 bytes，SHA256
`e8396e87b14ae16a7cca42484cc96d6f07be92288fc7cb7dc7c0639ad1d142cb`。
解包到独立项目目录后：

```bash
bash scripts/repair_v8_autodl.sh prepare
bash scripts/repair_v8_autodl.sh run
```

`prepare` 需要 Python 3.12 和依赖下载网络，建立单独 venv 并重跑 CPU 检查；包内未装入
依赖 wheels、venv 或模型。`run` 还需要可用 CUDA GPU 和 `compute-sanitizer`，只运行
上游 before/fixed 资格控制。产物在 `runs/repair-v8-qualification-20261018/gpu-controls/`。
**这个包包含人类修复和评分用例，只能用于研究人员资格验证，不能挂载给修复模型。**
当前入口也不是运行模型生成代码的隔离执行环境。

完整记录见 [准备状态](evidence/repair-v8-qualification-batch1-20261018/readiness.json)、
[CPU 参考检查](evidence/repair-v8-qualification-batch1-20261018/cpu-reference-summary.json)、
[导入与匹配测试收集](evidence/repair-v8-qualification-batch1-20261018/cpu-preflight-summary.json)、
[冻结的资格计划](evidence/repair-v8-qualification-batch1-20261018/qualification-plan.json) 和
[移植检查](evidence/repair-v8-qualification-batch1-20261018/portable-smoke.json)。
主要入口是 [`research/repair_v8/qualify.py`](research/repair_v8/qualify.py) 和
[`scripts/repair_v8_autodl.sh`](scripts/repair_v8_autodl.sh)。
接下来先做这 4 题的 GPU 资格验证，再处理余下 8 题；正式可见/隐藏测试封存、编辑
接口公平性和模型实验仍未完成，不能据本轮 CPU 检查新增任何模型能力结论。

## PR 筛选完成：开发候选与来源归档 — 2026-10-18 UTC

**这一批本地 CPU 筛选已完成：检索去重 471 个 PR，深入审核 40 个，保留 14 个
静态候选；按固定 hash 和仓库分层选出 12 个开发候选，另留 2 个开发备用。**
这是下一阶段的开发题来源准备，**GPU 合格题仍为 0，正式测试集封存题数仍为 0**。
尚未运行 Gemma、GPU 复现或训练，也没有租用或启动 AutoDL GPU。

机器可读结果：

- [筛选汇总](evidence/repair-v8-pr-screen-20261018/screening-summary.json)
- [12 个开发候选与 2 个备用的完整记录](evidence/repair-v8-pr-screen-20261018/development-queue.json)
- [471 个检索结果的逐项处理账本](evidence/repair-v8-pr-screen-20261018/candidate-ledger.json)
- [人工式源码审核记录、根因分组和每题复现设计](experiments/cuda-repair-v8-pr-screen.json)
- [来源包与完整性回执](evidence/repair-v8-pr-screen-20261018/source-bundle-receipt.json)

### 本批开发名单

| 仓库 | 开发 PR | 缺陷类型 |
|---|---|---|
| Liger-Kernel，4 个 | [1298](https://github.com/linkedin/Liger-Kernel/pull/1298)、[1397](https://github.com/linkedin/Liger-Kernel/pull/1397)、[1344](https://github.com/linkedin/Liger-Kernel/pull/1344)、[1376](https://github.com/linkedin/Liger-Kernel/pull/1376) | 非均匀上游梯度；JSD 下溢；AttnRes 超过 32 个 block；GroupNorm 多维反传分配 |
| Flash Linear Attention，2 个 | [1103](https://github.com/fla-org/flash-linear-attention/pull/1103)、[1108](https://github.com/fla-org/flash-linear-attention/pull/1108) | RWKV7 尾部越界；Triton kernel 内调用 host helper 导致编译失败 |
| FlagGems，4 个 | [553](https://github.com/flagos-ai/FlagGems/pull/553)、[519](https://github.com/flagos-ai/FlagGems/pull/519)、[6727](https://github.com/flagos-ai/FlagGems/pull/6727)、[6560](https://github.com/flagos-ai/FlagGems/pull/6560) | addmm bias 广播；排序 dtype 截断；attention 布局契约；原地 renorm 漏处理尾部 |
| vLLM，2 个 | [15909](https://github.com/vllm-project/vllm/pull/15909)、[33326](https://github.com/vllm-project/vllm/pull/33326) | Eagle 多 block 索引；GDN padding 状态索引为 −1 时越界 |

备用是 Liger [1374](https://github.com/linkedin/Liger-Kernel/pull/1374)（TVD 相等点的零次梯度约定）
和 [1433](https://github.com/linkedin/Liger-Kernel/pull/1433)（fused JSD 投影结果过早降精度）。
**14 个候选全部保留在 development / development reserve，不能再充当正式未见测试题。**
旧 Liger #1490、#1461 已从新增题中排除；#1298 虽然根因不同，也共享旧 CE 依赖，
因此只适合这批开发验证，不宣称它提供全新的正式迁移证据。

分配使用唯一固定 seed `cuda-repair-v8-pr-screen-20261018-development-v1`，
对 `seed + 换行 + root_cluster` 计算 SHA256，在仓库内排序后按仓库优先级轮流取题。
没有比较多个 seed，也没有使用模型能否修好的结果。资格失败可以在模型评测前按
预定备用顺序替换并记录理由；模型失败不能成为换题理由。目前备用集中在 Liger，
因此若 FLA/vLLM 复现失败，不能保证仍能凑齐原计划的开发数量与覆盖。

### 筛选范围和暂缓原因

检索窗口为 merged 2025-01-01 至 2026-10-18 UTC。四个仓库各检索标题中的 fix / bug，
vLLM 额外要求 Triton；每个查询按创建时间倒序最多取 100 条，共八个查询，其中三个
触及上限。这是**有边界、偏向较新 PR 的筛选批次，不是仓库全量普查或代表性抽样**。
FlagGems 旧组织名导致搜索失败后，已根据官方仓库返回值改为 `flagos-ai/FlagGems`，
原检索计划和修订原因均保留。优先做这一批开发资格验证，不能据此宣称已有 60–80 个正式题。

471 项中，40 项审核了 PR 正文及 diff，2 项是已使用的旧开发题；另外 **429 项仍是
未深入审核的检索池**，不是被判定不合格的题。深入审核的 40 项中：14 个静态候选，
22 个暂缓，4 个因当前范围不符而排除。逐项理由在账本中。

主要发现：

- **完整模块预算会限制任务来源。** 用固定 Gemma-4-31B tokenizer 实测了归档候选
  的真实 merge-parent 模块。14 个候选的完整原始 JSON 响应为 785–5,742 tokens，
  官方修复为 934–6,609 tokens，满足原始 6,144 / 输出 8,192 的筛选上限。
  很多一行修复位于 10k–30k token 的大模块里，不能只开放 gold 修改的函数或几行来
  绕过完整重写基线。暂缓项多为 PR-head 预估，未将其冒充最终 parent 资格判定。
  最终 prompt、只读上下文、API 卡、测试和输出格式开销仍须在封存前单独验证。
- **同源修复不能重复计数。** FlagGems #6727 明确是 #6667 的 cherry-pick，以
  issue #5896 为同一根因；FLA #1108 的 KDA/GDN2 两处复制代码计一题；Liger #1397
  的三个后端实现计一题。FLA #1135/#1165/#1171 暂放同一保守 lineage 组，尚未证明
  它们是可独立计数的根因。
- **merge 不能替代契约检查。** FlagGems #960 只涉及 size-one 维度 stride 规范化，
  尚无足够证据证明它违反公开 API，因此归档但不入选。vLLM #45361 的 PR 正文称
  `torch.round`，diff 却使用 half-away-from-zero；#42739 正文与最终补丁对 inplace
  的描述不同。这些项暂缓，未据静态差异宣称 GPU 测试失败。
- **开发题包含 kernel 本体与 host/kernel 接口两类修复。** 布局 materialization、
  分配及编译 API 修复也会影响真实 GPU 程序，但结果必须分开报告，不能统称为模型
  新写 CUDA kernel 的能力提升。原有自定义 kernel 必须继续实际执行。

### 已归档与尚缺的资格验证

已对 15 个 PR 归档真实 merge commit 的第一父提交、修复提交、相关前后源码、
现有/新增测试、上游依赖声明及许可证，包含 14 个候选和暂缓的 FlagGems #960。
没有把 GitHub PR 当前 `base.sha` 当成缺陷前提交。许可证分别保留 Liger BSD-2-Clause、
FLA MIT 和 FlagGems/vLLM Apache-2.0 原文。完整性校验通过 **240 个归档文件**；
15 对 owning-module 源码通过纯语法解析，并生成了真实 parent/fixed 差异。

所有检索原始响应、40 个 PR 的元数据/diff、源码快照及回执已打包为
[`investigator-source-bundle.tgz`](evidence/repair-v8-pr-screen-20261018/investigator-source-bundle.tgz)，
逐文件读回校验通过；包级 SHA256 和文件清单有独立回执。这个包含人类修复、后续历史
及评分测试，**仅供研究人员使用，不能挂载进模型修复容器**。它也不是完整仓库 checkout。
额外离线负控制确认：同长度源码篡改、越界软链接、路径穿越、缓存篡改和重复根因均被拒绝。
所有上游文件仅作数据读取/语法解析，没有 import 或执行。

下一步是根据名单制作 CPU 侧任务包和原仓库复现入口，再上 GPU 做资格控制。
每题已有触发条件、独立 oracle、保留行为、测试线索和具体风险说明；尚缺的是
可运行的锁定环境与隔离 runner、无答案泄漏的问题说明、可见/隐藏测试的封存，以及
before 连续失败 / fixed 完整通过各三个新进程的独立验证。

优先处理以下不确定性，而不是直接开始 Gemma 跑分：

1. vLLM 两题是本批最高的集成风险：已有测试线索主要是 serving/Eagle/MTP 路径，
   尚无本地可运行的单卡回归。#33326 的上游复现使用四卡大模型；必须证明单卡也能
   走原仓库调用路径，否则排除主终点，不以抽出的 kernel 代替仓库修复。
2. FLA #1108 的上游回归在 Triton 3.4 运行，而报告的编译错误发生在 3.6；需固定
   项目支持的环境做真正的 before/fix 对照。Liger #1344 作者明确未做 GPU 验证。
3. FlagGems #6560 的触发受 autotune 选择影响。强制某个合法 block 的实验只能作为
   辅助，还需原始入口稳定复现。旧 FlagGems snapshot 的依赖也需实际安装验证。
4. 官方 fix 同样必须通过扩展域、原有行为及支持的 memcheck；未通过的题记录失败并
   在模型输出之前剔除。当前“14 个静态候选”绝不等于“14 个可靠实验题”。

采集工具是 [`research/pr_screen_v8.py`](research/pr_screen_v8.py)，支持 discover / inspect /
archive / verify / scope / finalize / bundle。原始工作目录为
`runs/repair-v8-pr-screen-20261018`。本次工作只增加 v8 筛选代码、记录和本节报告，
v1–v7 的源码、结果及已封存审计不变。

## 下一阶段设计：独立 GPU 缺陷修复、回归与诊断机制 — 2026-10-18 UTC

这是本次审核之后的新实验设计，**不是已封存或已启动的实验**。
机器可读协议在 [`experiments/cuda-repair-v8-design.yaml`](experiments/cuda-repair-v8-design.yaml)，
样本量敏感性计算在 [`evidence/repair-design-v8/power-sensitivity.json`](evidence/repair-design-v8/power-sensitivity.json)。
v1–v7 的源码、配置、评分及审计回执保持不变。

**主问题：在相同模型和资源上限下，更新的执行诊断与原子化编辑，能否提高未用于
方法开发的 GPU 缺陷的可靠修复率；收益是否伴随重复错误和原有正确行为受损的减少？**

主模型固定为现有、已验证的 Gemma-4-31B-it BF16，冻结权重、采样和版本。
本阶段不新增训练；RL/SFT 旧结果作为背景，不进入主实验因子。
12B 可作为预先选定任务上的补充验证，但必须在正式结果揭晓前决定是否启用，
且无论结果方向如何都报告；它不是参数规模的因果实验。

### 任务与外部有效性

先使用旧两题加 **12 个新开发缺陷、至少三个仓库**完成可行性验证。这些题永久留在
development。正式目标为 **80 个独立根因缺陷、至少四个仓库**，资源受限时 60 个是
当前确认性方案的最低规模；若少于 60 个或四个仓库，必须降为探索研究，或在正式
模型输出产生之前修改并重新封存设计，不能默默降低要求后沿用原主张。

优先从 Liger、Flash Linear Attention、FlagGems、vLLM 的已合并真实修复中筛选，
不足时再按预定顺序看 SGLang/ExecuTorch。仓库名单只是候选来源，不保证有足够
合格题。公开 PR 的检索窗口、去重规则和抽样顺序在采集时固定；按根因聚类，
同一修复的 backport、复制到别的仓库的同源实现、同一问题换 shape 都不能算新缺陷。
每个正式仓库至少 10 题、单仓库不超过 35%，最终按仓库等权汇总。

每题在模型评测前必须满足：

1. 固定 parent/fixed 提交、许可证、依赖与镜像；缺陷版在三个新进程中稳定复现，
   官方修复版通过同一套完整回归。故障不能只是依赖装错或当前硬件不支持。
2. 修改范围先由缺陷前的模块、问题描述、栈和调用关系确定，再检查该范围能表达
   合法修复。不能看完人类 diff 后只开放那几行，也不能因模型修不好再排除题目。
3. 模型补丁能回填原仓库，并运行真实调用路径及相关原有测试。
   只有缩减复现、没有仓库集成验证的题进入辅助集，不混入正式真实修复终点。
4. 问题说明、API 和输入域足够明确；隐藏人类补丁、后续 Git 历史及最终评分测试。
   公开源码可能进入预训练，必须承认这种污染风险；“未见”仅指未用于本次方法开发。
5. 在模型输出之前按仓库分层和固定 hash 顺序确定 dev/test；记录全部排除原因、
   资格验证工时和最终样本数。纯性能优化、新功能、合成 mutation 不冒充真实 bug。

正式集不是“挑 Gemma 能修的题”。可行性可以用开发集检查；正式题的资格只由
复现可靠性、范围、测试及资源要求决定。48 个算子图候选也不计作真实缺陷数量。

### 五个对照条件

| 条件 | 输出形式 | 后续反馈 | 当前代码 |
|---|---|---|---|
| A | 完整模块 | 成败类别与通过数 | 上一次已应用程序 |
| B | 完整模块 | 结构化执行诊断 | 上一次已应用程序 |
| C | 原子化局部编辑 | 成败类别与通过数 | 上一次已应用程序 |
| D | 原子化局部编辑 | 结构化执行诊断 | 上一次已应用程序 |
| E | 原子化局部编辑 | 原始程序的初始简略结果 | 每次重置原程序，独立重采样 |

主要比较 **D−A**；预先指定的成分比较为 **D−C**（更新诊断的增量作用）和
**D−B**（编辑形式的增量作用）。E 用于说明多次独立尝试的表现，属于补充比较。
交互项 `(D−C)−(B−A)` 用于探索二者是否相互影响，不保证样本量足以检验交互。

所有条件共享问题说明、完整可编辑模块、只读上下文、可见测试和同一份版本化 API
说明卡，防止“不给 baseline API 信息”制造虚假优势。首轮都只接收原程序执行的
简略结果，不含详细诊断；A/B 共享同一个首轮候选，C/D/E 共享另一个首轮候选。
因此主研究测量的是**第一次尝试之后的反馈更新**，不泛称所有初始诊断设计的效果。
每个逻辑分支都计入首轮完整成本，物理复用单独记账。

两种输出形式具有**完全相同的可修改权限**：最多三个预定 owning modules，允许
修改其中的函数/方法、内部签名及调用点、import、模块常量，允许新增不冲突的 helper。
公共接口、许可证、测试、harness、依赖和其他文件保持受保护。完整模块和局部操作
都用 JSON envelope，并经过同一最终源码范围/语法检查。
局部操作只机械应用，不帮模型补代码、修语法或选择正确算法。
原模块的完整响应必须能放入预算，避免 baseline 从起点就必然被截断。

这要求新版本扩展 v7；现有 v7 冻结 import/常量并禁止新增 helper，不能直接用于
该公平比较。需要先用稳定的合法修复验证两种接口都可表达，而不是要求复制人类补丁。

运行时合约也要按任务冻结：目标计算必须实际使用自定义 GPU kernel，禁止用高层库
或 CPU 替代；原仓库本来合法的 host 操作按清单保留，并用 before/官方 fix 验证。
不能把旧评测对 PyTorch 的笼统限制直接套到完整仓库，从而把正确修复误判为违规。
新仓库 runner 仍需重新通过隔离和运行时观察控制。

### 资源预算与停止规则

每缺陷每条件 **3 个生成种子，每 episode 最多 4 次候选、4 次可见评测**。
每次最多输入 16,384 tokens、输出 8,192 tokens；episode 输入总上限 65,536，
输出总上限 32,768；每次候选执行上限 300 秒。格式失败、非法修改、截断和编译失败
都消耗尝试，没有免费的“再修一下格式”。条件间只改变协议指定的两个因素。

首次可见测试全过即停止；否则用最后已应用候选，完全没有合法应用时保留原程序。
正式生成及选择全部完成并封存后才解封隐藏判分，不允许根据隐藏失败追加尝试。
基础设施错误只允许重试同一保存请求/候选，日志单列；不能给失败模型换种子。
若任务本身有基础设施问题，盲审后对所有条件统一处理并披露。

80 题对应 **1,200 个逻辑 episodes、最多 4,800 个逻辑候选**；共享首轮后最多
4,080 次独立生成。60 题对应 900 episodes、最多 3,060 次独立生成。
这是生成预算上限，不是 GPU 时长预测；制题、镜像、测试和编译可能成为主要成本。

相同上限仍不等于实际算力相同：记录输入/输出 tokens、逐请求生成时间、编译与执行
时间、排队、加载、显存及 cache 状态。绘制成功率—实际资源消耗曲线，预算前缀只能
使用当时已完成的候选，不能用隐藏分数挑最好候选，也不能给超预算生成的尾部答案记功。
从完整轨迹计算的资源曲线是描述性的，不能冒充重新运行的硬截止策略。

另在正式输出前按 hash 选定 24 题，单独重跑 A/D 的**硬时限成本实验**：并发 1，
相同专用推理/评测硬件组合；时间上限由开发集 baseline episode 成本第 75 百分位
向上取整到分钟后固定。计入推理、编译、工具执行；首次引擎加载单独报告，
超时后的结果不计成功。分别报告总耗时和 GPU 资源秒，不能把 token 直接当 FLOPs。
当前硬件组合和时限尚未校准，因此暂不能写具体小时数或上线提效倍数。

### 回归检查与成功定义

最终成功必须同时满足：**修复缺陷 + 保留原有正确行为 + 隐藏边界检查 + 原仓库集成
+ 内存/输入输出合约检查**。测试数量不代替覆盖质量，每个测试族需说明其排除的错误。

| 测试面 | 设计 |
|---|---|
| 原缺陷 | visible 有可复现触发；hidden 用不同 seed/shape 的同类触发 |
| 原正确行为 | 保存缺陷版已经通过的测试，检查补丁是否将 pass 变为 fail |
| 布局与索引 | 合法的非连续/独立 stride、storage offset、mask 与 padding |
| 尺寸 | 1、奇数/质数，以及 tile 边界前一位、边界、后一位 |
| 数值与 dtype | 域内的零、常量、正负数、小值、大有限值；支持的 FP32/FP16/BF16 |
| oracle | 独立稳定公式/实现，必要时 FP64；不能只与人类补丁相同就认为正确 |
| 集成 | 在原仓库 pinned revision 的真实调用路径运行补丁与保留测试 |
| 内存和合约 | 支持的 Compute Sanitizer memcheck、输入不变、输出存储/shape/dtype 合约 |

只测试规范支持的域；NaN/Inf、空张量、负 stride 等不自动视为合法输入。
适用轴、组合覆盖、随机种子及算子/dtype 容差在模型生成前冻结。官方修复也必须
通过扩展测试；另用若干已知错误补丁验证测试确实能发现问题。Sanitizer 不提供
所有 race 或内存行为的完备证明，能力边界按版本记录。
依据是 [NVIDIA 的工具说明](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

特别将旧 31B 的 tanh 溢出纳入开发检查：大幅度输入在域内时必须覆盖。
新正式测试在封存后若发现额外边界问题，另作追加审计并报告，不能悄悄更改主评分。
格式/编译/崩溃造成的不可观察结果与已执行的语义回归分开记录。

### 如何支持机制解释

主矩阵全程记录：编辑应用失败、格式/编译/数值失败、可见通过后隐藏失败、
原正确测试被破坏，以及标准化错误签名再次出现。签名按异常类型/API/函数或数值
失败测试族定义；“同一签名”不是已证明“同一根因”，这两个词不能混用。
全部可执行候选的保留行为面板可在生成完全冻结后离线评分，不反向驱动策略。

再增加一个**同状态单步分叉实验**：从 A/B 共享首轮的失败状态中，每个缺陷按
预定顺序取最早的一个 scope-valid 编译或数值失败，按固定 hash 最多选 24 个缺陷。
不能按后来哪个策略成功来选，也不能为了凑数换更难种子。三条分支使用同样的当前
代码、编辑接口和一次生成预算，各重复三次：

- 仅 verdict/counts；
- 完整诊断；
- 诊断中删除一个按观测签名预定的字段组，例如 API 异常、shape/stride 或数值误差。

看下一步是否消除观测错误、完成可靠修复，以及是否破坏原正确行为。
这最多新增 216 次生成。若合格状态较少，直接报告实际数量；不可分类状态保留在
前两项比较，并标注字段删除不适用。它是小规模、预先规定的补充实验。

完整诊断胜过 verdict、删除相关字段后收益下降，才支持模型利用了该执行信息；
D−B 若同时表现为更少结构错误/保留测试破坏，则支持局部编辑减少附带修改的解释。
这些是行为层面的干预证据，不等于神经机制、形式化因果中介或训练迁移机制证明。

### 统计与可接受的结论

先对每题三个种子的成功指示取平均，再在仓库内对独立根因缺陷取平均，最后仓库等权。
统计单位是根因缺陷，同题所有条件和种子作为一个整体重采样。
D−A、D−C、D−B 三项构成预先指定的检验家族：Bonferroni 双侧总体 α=0.05，
报告 98.333% 分层 task-cluster bootstrap 区间和零假设中心化的 studentized
bootstrap p 值；100,000 次重采样，尾部采用 plus-one 估计，退化样本报告不可检验。
仓库视为固定，不宣称四个仓库就是大量独立仓库样本。

这仍依赖仓库内根因近似独立、任务代表性和可估计方差，属于近似推断；没有
“无需任何分布假设”的保证。另报每仓库效果、leave-one-repository-out、全部失败
和排除记录。只有多个仓库保留改善且总体不是由单一仓库驱动，才写跨仓库证据；
不要求每个小仓库单独 p<0.05，也不把总体显著偷换成每个仓库都有效。

10 个百分点是实际意义的规划目标。点估计超过它不代表已经证明至少改善 10 点；
后一个说法需要相应区间下界也超过 10 点。若 D−A 有收益而成分比较不明确，只能
主张整体方案有效，不能宣称两部分都得到验证或证明了协同作用。

样本量计算使用假设性的配对差异模型：单种子 discordance 为 q、差值为 δ，
三个重复之间相关为 ρ，则每题平均差异的方差为
`(q − δ²) × (1 + 2ρ) / 3`。以下是正态近似的规划敏感性，**不是现有实验结果，
也不是未来分层 bootstrap 的功效保证**：

| 真差值假设 | q | ρ | 60 题近似功效 | 80 题近似功效 | 80% 功效所需题数近似 |
|---|---:|---:|---:|---:|---:|
| +15 个百分点 | 0.30 | 0.50 | 62% | 77% | 87 |
| +15 个百分点 | 0.50 | 0.50 | 37% | 49% | 149 |

因此 60–80 题是可行性与证据强度的目标区间，不是显著性承诺。如果资格验证显示
能准备更多题，应在正式输出之前依据开发成本/方差与预定效应重新封存规模；
不能看正式 p 值后继续补题。多跑种子不能替代更多独立缺陷。

### 贡献定位与执行顺序

“执行反馈 + 编辑接口”本身已有 [SWE-agent](https://arxiv.org/abs/2405.15793)
等工作；真实 PR 与框架内评测也已有
[RealisticTritonBench](https://arxiv.org/abs/2608.12004)，其任务同时包含优化、修改与
新 kernel，并不是 31 个可直接复用的缺陷修复题。
[GEAK](https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html)
也已有执行迭代和失败分析。该项目的贡献必须来自严格实验发现，例如 GPU 诊断的
信息利用、可见测试与真实回归的差距、编辑方式对错误恢复的影响及其任务边界，
不能仅以组合这些组件声称新颖。

优先执行顺序：**12 个开发缺陷资格验证 → v8 接口/回归/成本记录集成 → 新 GPU
控制与开发可行性 → 封存正式任务和统计协议 → 完整矩阵与机制分叉 → 隐藏评分及发行包**。
开发阶段先运行五条件、每题两个种子；进入正式阶段的门槛是全部控制通过，至少
三个新缺陷被某种方法可靠修复，且至少三个缺陷没有在所有条件/重复下全过，避免
完全地板或天花板。门槛不要求 D 已经胜过 A。未通过时只用开发集诊断、修改并重新
验证，不查看正式集输出，也不把开发集转入 test。
当前没有合格正式任务清单、v8 runner、扩展回归的 GPU 验证和已校准成本时限，
所以这个文件是可审阅的设计，不是可直接启动的控制器；本次没有启动下载、训练或远端评测。

## 全量审核：已有工作、数据与结论边界 — 2026-10-18 UTC

本节是截至本次审核的汇总结论；下方按时间倒序保留历史记录。
审核基线为 `7462baf`。完整盘点和重算输出在 `runs/full-audit-20261018/`，
可跟随 Git 保存的摘要、校验记录和测试结果在
[`evidence/full-audit-20261018/`](evidence/full-audit-20261018/)。
独立审核入口为 [`research/full_audit_20261018.py`](research/full_audit_20261018.py)。

**判断：主要实验记录完整、核心数字可重建；旧口径的数值正确性收益成立。
但“自定义 CUDA/Triton 编程能力提升”“RL 优于 SFT”“诊断提高未见缺陷修复率”
目前均未获得足够证据。最值得继续验证的正向线索是提供 host 脚手架，以及
一次诊断驱动的 API 修复；它们仍属于开发实验。**

这里的“未见”需要区分：原 Level 2 未参与梯度训练，但研究者在早期 pilot 中
已观察过这些题的结果。R2/R3 是新生成样本和训练种子的复评，不是研究者未见的新任务验证。
本次审核覆盖存量清单、各主要实验的数据链、关键训练/评测/统计/修复代码和 CPU 检查；
没有重训模型，也没有重新执行全部 GPU 程序。候选代码仅作文本/AST 检查，没有在本机导入执行。

### 1. 已有资产与核验范围

| 对象 | 本次核验结果 | 解释 |
|---|---|---|
| 存量文件清单 | 106,394 个文件，46,459,725,839 字节 | 排除环境、缓存、Git、软链接及本次审核输出；包含多份重复备份，不代表独立实验数量 |
| 压缩归档 | 50 个，共 41,235,426,647 字节，全部重新计算 SHA-256 | 32 个自动匹配到相邻历史 JSON 回执；余下 18 个未被这一回执扫描规则匹配，不能一概称为损坏或已有独立回执 |
| 初始 pilot 主归档 | 7,143,496,613 字节，历史 SHA-256 一致 | 完整流式读取 11,751 个成员，逐个哈希 8,486 个普通文件 |
| R2/R3 主归档 | 30,779,004,695 字节，历史 SHA-256 一致 | 完整流式读取 120,315 个成员，逐个哈希 86,264 个普通文件 |
| 9 个最终微调模型 | C/S/SFT 各 3 个；归档内 adapter 哈希与最终评测请求全部一致 | 中间 checkpoint、调参模型及重复备份不计为额外训练种子 |
| 正式终评 | 10 个策略 × 120 题 × 8 次生成 = 9,600 条 | 检查网格、请求/完成回执、生成与评分行对应关系、参考源码和 adapter 身份 |
| R3 完整统计报告 | 从原始行重建，与归档 JSON 的全部字段一致 | 包括各策略区间、源码类别、主要比较及共同正确样本上的速度比 |
| 后续 6 批研究 | v3、v4 接口、v4 严格复评、v5、v6 12B、v6 31B | JSON 和 Markdown 报告均逐字重建一致；归档与本机源码封存哈希一致 |
| CPU 检查 | 含 PyTorch 的本机环境中 179 项通过 | 覆盖原 tests 和全部后续研究测试；未重跑 GPU 控制实验 |

两份主归档的 SHA-256 分别为
`d58e5119237fe80eeb3553d3d9fe800f30ce75f6ad6d0509bba7d28bff1e96fc` 和
`8cc3de5ecadc11e6ce5339dacba4bafc702e0f79d4892f57f927cb07b9f01e74`。
完整读取中未发现重复成员名或链接成员；在已执行的哈希、网格、统计重建检查中未发现不一致。
这说明现存备份可读取并能支撑报告，不等于证明历史上每个运行时判断都绝对正确。

CPU 测试使用 `/home/noname/anaconda3/bin/python3`，PyTorch 为 `2.11.0+cu130`。
开发专用 `.venv` 没有 PyTorch：同一命令在该环境得到 176 通过、2 跳过、1 失败，
失败发生在一个研究 runner 测试的 PyTorch 导入。因此不能只运行默认 `pytest` 就声称
全部研究检查通过，也不能把这个依赖缺失当作历史模型实验失败。

本次 SSH 只读探测超时。最后可验证的远端终态仍是 2026-10-17 的 31B 完成记录；
已拿到本地完整归档。没有本次实时 GPU、磁盘、平台开关机或账单证据。
远端关机动作缺少返回回执，不能仅由 SSH 不通认定机器已关机。

### 2. 实验总账：哪些完成了，哪些只是候选/代码

| 阶段 | 已有规模 | 核心结果 | 证据地位 |
|---|---|---|---|
| C0–C6 初始 pilot | 9 个 base/中间/最终策略 × 120 题，共 1,080 程序 | 最终 C/S 的数值正确性高于 base | 探索；C 在 step 35 停止，S 到 step 40 |
| R2 独立生成复评 | base + 两个 pilot 模型，各 960，共 2,880 | 重算指标及配对比较一致 | 同一批已知任务，新生成种子 |
| R3 三种子确认 | base + C/S/SFT 各 3，共 9,600 | 旧数值口径收益保留；C 对 SFT 无明确优势 | 已完成；最强的微调证据 |
| 存储事故复评 | L1Norm 80 个原程序/原输入种子 | 40/80 → 61/80；恰好 21 条 crash → correct | 基础设施诊断，不是新生成或模型进步 |
| v3 新图开发试验 | 6 图 × 2 prompt × base/C/SFT 首种子，共 36 | 严格成功 0/36 | 开发试验，不是 48 个新候选图的正式评测 |
| v4 同程序严格复评 | 100 个 L2 × 8 × base/3C/3SFT，共 5,600 | 严格自定义 kernel 成功 0/5,600 | 对旧输出增加约束；不是严格 prompt 下的新生成 |
| v4 host 接口试验 | 6 图 × 7 策略 × 2 接口，共 84 | 全程序 1/42；固定 host 16/42 | 明确的开发信号，仍缺独立任务确认 |
| v5 一次修复与诊断形式 | 2 缺陷 × 7 策略 × 3 反馈 × 2 重复，共 84 | base 的 grouped 条件 2/4；其余条件无成功 | 两个同仓库已见开发缺陷 |
| v6 12B 迭代修复 | 12 episodes，37 次独立生成 | 重采样 1/4；verdict 1/4；详细诊断 0/4 | 开发；12B 完成，父控制器因 31B 启动资源不足停止 |
| v6 31B 续跑 | 12 episodes，22 次独立生成 | 重采样 2/4；verdict 2/4；详细诊断 3/4 | 开发；一个额外诊断成功 |
| 新任务候选库 | 48 个算子图，6 个 family | 有生成器和参考代码 | 未封存、未进行模型正式终评；AST 不同不等于语义独立 |
| 真实修复候选库 | Liger 2 个 PR、ExecuTorch 1 个 PR | 仅 Liger 两个完成缩减复现与模型试验 | 还没有跨仓库修复结果 |
| v7 受限函数编辑 | 编辑器、作用域校验、14 个 CPU 测试 | 原型可检查/应用函数替换 | 无模型响应协议、完整 runner 或 GPU 模型结果 |

不能直接将各行相加作为独立样本数：R3 复用了 R2 的 960 条 base 评测；
5,600 条严格复评和 80 条存储复评都是旧程序重测。v6 三种策略还共享首轮候选。
重复输入、训练种子、生成次数、同一缺陷的 replicas 与独立任务必须分别计数。
初始 C4 的 prompt/task 选择，以及 SFT 的学习率选择，属于 TRAIN 调试成本，
不能作为额外 held-out 改善证据。C0–C6 PASS 表示各操作门槛在当时设置下完成，
并不意味着所有科学假设成立，也不是隔离机制或反作弊机制的普适证明。

### 3. 微调究竟提高了什么

正式主终点是 100 个 Level-2 任务的原始数值正确率，每题每模型 8 次生成；
C/S/SFT 先分别在三个训练种子上取平均。C 为正确性奖励，S 为带速度项的奖励。

| 策略 | Level-2 数值正确率 |
|---|---:|
| base | 54.875% |
| C，三种子平均 | 72.000% |
| S，三种子平均 | 71.667% |
| SFT，三种子平均 | 71.250% |

| 主要比较 | 差值，百分点 | Bonferroni 调整的 98.333% task-bootstrap CI | 事后双侧 bootstrap p |
|---|---:|---|---:|
| C − base | +17.125 | [12.625, 21.792] | 约 0.000010 |
| S − base | +16.792 | [12.333, 21.417] | 约 0.000010 |
| C − SFT | +0.750 | [−1.958, 3.583] | 0.518535 |

p 值来自另外保存的、以任务为单位、零假设中心化的 studentized bootstrap，
100,000 次重采样；前两项是零次尾部命中后的 plus-one Monte Carlo 分辨率下限，
不是精确概率。三项 Bonferroni 校正后分别约 0.000030、0.000030、1.000000。
这些 p 值是事后辅助分析，不能冒充预注册检验。主要区间和 p 值都以这批已训练
checkpoint 为条件，并依赖任务重采样假设，不覆盖任意训练种子、仓库或任务分布。
“没有发现 C 优于 SFT”不等于二者统计等效，也不能证明 RL 没有潜在优势。

所有正式微调实际只使用 **`level1/40_LayerNorm` 一个任务**。
数据划分中存在 80 个 TRAIN 任务，不代表训练使用了这 80 个。
SFT 是同一个 Gemma-4-12B-it base 的成功样本筛选微调：每种子生成 320 个候选，
分别接受 171、177、200 个，保留重复频次；不是大模型教师蒸馏，也不是只学一个答案。
另有 TRAIN 上三档学习率调参，选出 `1e-5`；额外生成、筛选和调参成本需要计入。

9 个最终模型中，8 个到 step 40，`S-20261021` 因 detector finding 在 step 15 停止，
终评保留的是这个实际停止的策略。原 detector 的 NaN/`torch.equal` 行为可能造成
误报，但历史原始张量已不全，不能把该次触发的原因追认成已证明的误报或攻击。
C 与 S 还同时改变了失败惩罚：C 的执行/格式失败为 −0.1，S 为 0；因此它们不是
单独增添速度项的干净消融。候选数量上限接近，也不等于实际 token、FLOPs、GPU 时间相同。

**速度方面没有稳定的广泛收益。** Level-2 上，各种子相对 base 的共同正确样本
几何平均速度比，C 为 0.9885/1.0013/1.0002，S 为 1.0112/0.9901/0.9991，
SFT 为 1.0038/1.0005/0.9882。它们是条件描述指标，没有证明总体更快或统计等效。
成功集合变化、计时噪声和原代码允许调用 PyTorch 都限制了速度解释。
“CUDA 会自动优化，所以没有加速”也不是本项目已经验证的机制。

### 4. 严格评测为什么改变结论

旧 prompt 允许 PyTorch 实现；后来严格口径要求真实执行自定义 CUDA/Triton kernel，
并限制 host 端数学运算。因此旧数值收益成立与严格成功率为零可以同时成立。

5,600 条严格复评中，3,878 条观测到数值正确，491 条数值错误，1,231 条数值结果
不可观察。只有 1 条满足运行时 kernel 合约，而它数值失败；没有一条同时满足两者。
主要结果类别为 4,236 条 contract violation、949 crash、264 format、133 incorrect、
13 source blocked、5 sandbox blocked。类别有优先级，不能与各个布尔观测直接混用。
不同版本对缺失观测还存在 `False`/`None` 编码差异，跨版本汇总必须单独处理。

这项复评足以限制论文措辞：**不能把旧数值正确性收益直接称为“写自定义 CUDA
kernel 的能力增强”**。它不证明旧实验作弊，不说明 5,600 个程序数值全错，
也不代表同一模型在严格新 prompt 下必然全部失败。后者只有另一个 36 程序的
小型开发试验，尚无独立新任务终评。

21 条 ENOSPC 事故发生在 Level 1 的 `38_L1Norm_`，同源码/同输入种子复评恢复为正确。
它们应在基础设施失败分析中披露；不影响上述 Level-2 主比较。原始评分未覆盖，
复评变更了存储/评测实现，不能用其计时推断模型加速。

### 5. 接口与真实修复：正向线索及本次新增问题

**host 脚手架确实值得研究，但当前不是纯编辑接口消融。** v4 给模型固定的 host
包装与 launch，改变了模型需要负责的代码范围。全程序 1/42 对固定 host 16/42，
分策略为 base 0/6→2/6、C 0/18→6/18、SFT 1/18→8/18；只有六个手写开发图。

本次静态复核发现，全程序 24 次 format 失败中，20 次未能提取程序：19 次是实际
Python 语法错误，集中在 kernel launch 表达式，例如 `run_kernel[grid=(M,)]`；
另 1 次没有闭合代码围栏。其余为 3 次字面 ABI 检查失败和 1 次重复定义 ModelNew。
这支持“host/launch 是当前瓶颈之一”的开发观察，不能泛化成抽取器误杀了有效程序，
也不能据此证明数学推理机制得到改善。

**发现一项需要披露的口径差异：** 固定 host 的 `C-20261023/reverse` 输出达到
4,096-token 长度上限，但包含可接受的函数，旧 runner 仍评为 correct。
如果事后敏感性分析把所有非 `stop` 输出都计为失败，结果是 **1/42 对 15/42**。
原始 16/42 保持不变；v5/v6 已采用不同的截断检查，不能跨版本无说明比较格式成功率。

真实修复材料来自 Liger PR 1490/1461 的缩减复现。内核函数按原文件复制，
wrapper、依赖和输入做了记录在案的改编；before 失败/after 通过的 GPU 控制已留证，
向模型隐藏 PR 定位信息及 withheld 测试。它们是有真实来源的函数级修复，
还不是完整仓库导航、定位、修改和集成测试的 SWE agent 基准。
ExecuTorch PR 17832 仅有源码候选归档，没有资格验证和模型结果。

v5 的两个成功都来自同一个 independent-strides 缺陷的 base/grouped 条件，
不足以声称诊断普遍有效或微调损害修复。v6 的 31B 结果中，case-002 的两个首轮
成功在三种策略间共享；7 个成功 episodes 实际只有 **3 个不同成功程序**。
新增的那个是 case-001/replica-1 在详细诊断的第 4 次尝试：替换不可用的 `tl.tanh`
后通过当前 visible/withheld 测试。22 个候选中 17 个收到这个缺失 API 诊断，
说明反馈经常未能使模型消除同一错误；没有证明反馈已提供全部必要信息。

**本次确认了该新增成功的数值边界风险。** 候选使用
`(exp(x)-exp(-x))/(exp(x)+exp(-x))` 实现 tanh。
对同一公式作人工转写的 CPU float32 探针，在 ±90、±100 处产生非有限值，
而 `torch.tanh` 有限。现有测试主要是标准正态输入，没有覆盖这个大幅度区间。
这是独立的算术风险证据，**不是已经执行的 GPU 候选反例，不回写原成功评分**；
在补充 GPU 边界回归前，不宜称它为全输入域可靠修复或“没有新增回归”。

31B 三策略各用了 10 个逻辑尝试，但 prompt token 分别为 58,856/28,486/28,936
（重采样/verdict/详细诊断），输出为 17,271/13,013/13,198；相同尝试上限不等于
同一实际算力。12B/31B checkpoint 也不只参数数量不同。不能由这两题推导
诊断的平均收益、模型缩放规律、延迟收益或 RL/SFT 优势。

**v7 目前还没有实验结果。** 其函数编辑原型会以
`change_outside_editable_functions` 拒绝这个保存的 31B 成功程序，因为它新增 helper
并改写了函数以外的结构。两种接口若用同一限制并不因此不公平，但必须先验证
作用域仍能表达所需修复；不能直接拿 v6 成功率当 v7 对照，也不能声称局部编辑已有效。

### 6. 工程与复现还缺什么

1. **论文主张与评测目标仍有断层。** 最大规模的可靠数据证明的是宽松口径数值收益；
   SWE/严格 GPU 修复方向只有开发规模，尚无未见缺陷、跨仓库确认和机制消融。
2. **独立测试集没有完成。** 48 新图候选未封存/终评；真实缺陷只有同仓库两题。
   反复观察过的这两题不能再充当最终未见测试集。隐藏同题输入不等于隐藏新任务。
3. **回归覆盖仍不足。** 包括数值幅度、布局、边界尺寸、dtype 和原仓库行为范围。
   减少重复错误/新增回归的机制结论还需要统一定义与完整轨迹统计，不能挑一条成功说明。
4. **公平成本尚未统一。** 训练/筛选/调参、提示长度、生成、编译及执行成本分散在记录中。
   现有 matched-attempt 实验可以描述，但不能自动称为同算力或生产提效。
5. **公开复现包尚未完成。** 本地原始 runs、上游候选源码、checkpoint 主要位于 ignored
   目录或大归档，Git clone 本身不能重建结果。需要按许可证、模型使用条件整理稳定下载、
   哈希、依赖、任务材料和从原始记录到表格的单一入口。
6. **安装与测试入口有缺口。** `pyproject.toml` 的包发现只包含 env/train/eval/tools，
   未包含 tasks/research；从 checkout 运行可用，安装 wheel 后不应假定研究入口存在。
   默认 pytest 只发现 tests，遗漏研究目录。GPU 运行栈与本机 CPU 栈也应分别标明。
7. **文档曾严重滞后。** 本次已更新 README，移除“C1–C3 尚未实现”等旧状态，
   并限定数值正确性主张。原 `REPORT.md` 只记录最初 C6，不能作为整项目最终报告。

为保持历史可审计，本次没有修改已封存训练/评测源码、配置、切分、候选程序或评分。
安装入口、回归套件和作用域设计应在新版本解决，随后重跑相应控制；不能把新规则
悄悄套回旧结果后仍称为原实验。数据完整性与任务有效性是两件事：前者本次通过
多项核验，后者仍受上述协议和测试覆盖限制。

### 7. 当前可写入论文的结论与优先级

可以写：在这个单任务训练、已知 KernelBench 任务和保存的三种子设置下，RL/SFT
微调提高了允许高层算子的数值正确性；未发现 C 相对 rejection-sampled SFT 的明确额外
收益，也未建立广泛运行时加速。对相同旧输出增加自定义 kernel 合约后，旧收益不能
解释为严格 kernel 成功。host 脚手架及诊断引导修复给出值得进一步验证的开发观察。

目前不能写：RL 优于 SFT；自定义 CUDA 编程普遍增强；诊断在同算力下显著提高未见
GPU 缺陷修复率；收益跨仓库保留；已减少新增回归并解释了迁移机制。
现有材料能形成诚实的实证报告，但还没有闭合一篇以“可靠 GPU 修复 agent”为主张的
优秀投稿证据链；是否获奖更不能从现有成功个例推断。

后续优先级建议：先在新版本完成数值边界测试和可表达修复的作用域控制；再固定
诊断、编辑范围、尝试/token 预算及独立任务资格规则，完成多个仓库的新缺陷评测；
最后报告配对任务效应、失败迁移和成本曲线。现有大规模宽松评测作为背景与反例保留。
这比继续扩大旧 LayerNorm RL/SFT 比较更直接补上当前证据缺口。本次审核未启动新实验。

### 8. 重建入口

从包含既有本地数据/归档的 checkout 执行，输出必须是新的审计目录；不需要 GPU：

```bash
.venv/bin/python -m research.full_audit_20261018 assets --out runs/full-audit-recheck
.venv/bin/python -m research.full_audit_20261018 archive-members --out runs/full-audit-recheck
.venv/bin/python -m research.full_audit_20261018 studies --out runs/full-audit-recheck
.venv/bin/python -m research.full_audit_20261018 legacy --out runs/full-audit-recheck
.venv/bin/python -m research.full_audit_20261018 final-report --out runs/full-audit-recheck
/home/noname/anaconda3/bin/python3 -m research.full_audit_20261018 probes --out runs/full-audit-recheck
```

最后一行需要含 PyTorch 的解释器；上述绝对路径是本次工作站环境，不是可移植依赖声明。
assets 是运行时文件盘点，后续新增审计目录/文档会改变其文件总数；实验归档与评分哈希
应保持一致。重建脚本依赖当前本地备份位置，不能替代尚未完成的公共数据发行包。
主要证据包括 `archive-hashes.json`、`main-archive-members.json`、
`study-reconstruction.json`、`legacy-and-training.json`、`confirmation-report.json`、
`interface-format-diagnosis.json`、`numerical-and-scope-probe.json` 以及 CPU JUnit 回执。

## 31B complete: one additional diagnostic repair; archive audited — 2026-10-17 UTC

Controller `2026-10-17T204539.738896_0000` completed without an error at
**21:07:10 UTC**. Its repair phase took approximately **20 minutes** (about
21.5 minutes including fresh startup controls), completing all 12 episodes
with 22 distinct generated candidates. The verified results are:

| Strategy | 12B resolved / episodes | 31B resolved / episodes |
|---|---:|---:|
| Independent resampling | 1 / 4 | 2 / 4 |
| Revision with verdict/counts | 1 / 4 | 2 / 4 |
| Revision with detailed diagnostics | 0 / 4 | 3 / 4 |

Each strategy covers the **same two previously seen defects, with two replicas**.
A resolved episode passes both visible and withheld strict regression tests.
31B solved the independent-stride defect (`case-002`) on both first draws.
Those first draws are shared across all three strategies: the six logical
successes represent only two generated programs, not six independent repairs.

The additional success is `case-001`, replica 1, detailed-diagnostic revision,
on its **fourth attempt**. Independent resampling and verdict-only revision
failed that same task/replica. Its preceding attempts received the bounded
compiler diagnostic that `triton.language.tanh` is unavailable. The final
program replaced the `tl.tanh` calls with a custom Triton implementation and
passed both visible and withheld tests. Across all 22 candidates, 17 received
this missing-API diagnostic; repeated feedback often failed to remove the API.
Overall physical outcomes were **3 correct, 18 crashes, 1 numerical failure**.
The 7/12 logical success count therefore represents **three distinct successful
programs**, not seven independent discoveries.

This is a concrete development lead for execution-guided API repair, not an
established average diagnostic effect or a causal parameter-count result. It
does not establish RL/SFT superiority; no new training occurred. The successful
custom tanh expression also merits numerical-range review: its exponentials
can overflow for large absolute inputs. This is a source-level concern, not an
additional executed test or a retrospective change to the frozen scores. Passing
the current regression suite is not a claim of correctness over every input.

The **1,158,235-byte** archive was locally verified at **21:07:59 UTC**, SHA-256
`c20870fa852a1046590d06ab013e136257679d06589849fc592f0aac29c8ead5`.
Local audit reproduced both 12B and 31B reports byte-for-byte, checked their exact
episode grids and stopping/selection traces, and verified all 131 original /
137 continuation source files. Both generations and failures remain retained.
The controller was configured to shut down after backup acknowledgment. SSH is
now unreachable and the monitor has exited, but no shutdown-return receipt was
captured; physical power and billing remain unverified. No experiment is running
according to the last terminal record.

Evidence: `evidence/diagnostics-v6/resume31b-complete-audit-20261017.json`.
Reproduced report: `runs/diagnostics-v6-resume31b/terminal-audit-20261017/gemma-4-31B-it/REPORT.md`.

## 31B continuation running; GPU cache issue cleared — 2026-10-17 UTC

The authorized endpoint is now `connect.westd.seetacloud.com:31181`, on host
`autodl-container-6440418da2-103a5874`. The existing SSH key works and the host
key matches the previously authorized instance. The GPU is an RTX PRO 6000
Blackwell Server Edition with 97,887 MiB VRAM; approximately 301 GiB disk space
remains free. CUDA execution and the full 137-file continuation source identity
passed validation, using the existing preparation without changes.

Controller `2026-10-17T204539.738896_0000` started at **20:45:39 UTC**, PID 3423.
All six fresh isolation probes, eight original-fail/fixed-pass controls, and the
bounded diagnostic-feedback control passed. Full model verification completed
before first generation. The four first-round requests match the stopped parent
run exactly, apart from the preparation directory path; prompts, order and
sampling seeds are unchanged.

The **31B engine initialized successfully at 20:48:45 UTC**. BF16 weights use
57.76 GiB; with memory utilization 0.90, available KV cache is **24.83 GiB**
(29,562 tokens), clearing the previous 20.64 GiB requirement for a 24K request.
At **20:49:20 UTC**, first-round generation was active, GPU utilization was 98%
and memory use 88,379 MiB. No completed generation or model score had yet been
observed at that snapshot; this establishes startup viability, not repair gains.
At **20:50:43 UTC**, the first two of four initial candidates had been generated;
no candidate had yet been scored. Generation was continuing normally.

This runs only the previously ungenerated 31B arm: 12 development episodes,
at most four attempts each, with independent resampling, verdict revision and
detailed-diagnostic revision. Completed 12B outcomes remain retained. It is not
new training, a formal test, or the separate scoped-edit prototype.

The local `cuda-gemma4-diagnostics-v6-resume31b.service` monitor is active on the
new endpoint and restarts on failure. The remote controller retains bounded
stage timeouts, reporting and archiving, local SHA-256 backup acknowledgment,
and the authorized shutdown after completion or failure. No new deployment or
experiment source change was required for this GPU launch.

Evidence: `evidence/diagnostics-v6/resume31b-gpu-start-31181-20261017.json`.

## Storage reclaimed; 31B continuation deployed on CPU boot — 2026-10-17 UTC

The endpoint `connect.westd.seetacloud.com:21785` is reachable in **no-GPU mode**.
User-authorized storage cleanup reclaimed **65.9 GiB**. The 470 GiB data disk now
has approximately **301 GiB free**, with usage reduced from 51% to 36%.

- Archived C3 tensors were hashed before deduplication. **101 duplicate files**
  now share storage with matching canonical files, saving **37.3 GiB**. All 111
  original tensor paths and byte contents were retained and checked again after
  the operation. Ownership and permissions match within each group. The C3
  directory occupies approximately 4.1 GiB instead of 42 GiB. Treat these
  completed, hardlinked artifacts as immutable; new gate runs use fresh paths.
- The **28.7 GiB** remote replication archive was removed only after both its
  remote and existing local copies were rehashed to
  `8cc3de5ecadc11e6ce5339dacba4bafc702e0f79d4892f57f927cb07b9f01e74`.
  The verified local copy remains at
  `runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/evidence-and-checkpoints.tgz`.
  A `storage-relocation.json` receipt records this at the original remote
  controller directory; the historical `archive.json` remains unchanged.
- All model weights, training checkpoints, original experiment records, and
  sealed source files were retained. The new v6 parent archive is also retained.
  No generated candidate code was executed during cleanup.

The six-file 31B continuation package has now been deployed. Remote preparation
`runs/diagnostics-v6-resume31b/prepared-20261017` passes source, protocol, model
metadata and file-size validation; its receipt SHA-256 is
`6e0e3b2af0a440154302a5226530cb5b26c0a80a95c0d1ba06e581c4ce62af7f`.
The full 137-file continuation source seal matches the workstation. Memory
utilization is 0.90, with the original BF16/context/output settings preserved.
No GPU controller or inference was launched. GPU controls and complete model
weight hashing remain required at the next GPU launch.

Evidence: `evidence/storage/cleanup-20261017.json` and
`evidence/diagnostics-v6/resume31b-remote-ready-20261017.json`.

## 12B finished; 31B startup blocked; continuation prepared locally — 2026-10-16 UTC

Controller `2026-10-16T222119.856430_0000` stopped at **22:58:42 UTC**.
The 12B arm completed all 12 development episodes, producing 37 distinct
candidates. Both successes passed visible and withheld strict regression tests:

| 12B strategy | Resolved episodes | Logical attempts |
|---|---:|---:|
| Independent resampling | 1 / 4 | 14 |
| Revision with verdict/counts | 1 / 4 | 15 |
| Revision with detailed diagnostics | 0 / 4 | 16 |

Both successes concern `development/case-002`, the independent-stride defect.
The other defect remains unresolved. There are only **two previously seen
development defects**, with two replicas per strategy; these observations do
not establish a benefit or harm from rich feedback, or a general repair rate.

The 31B engine loaded its BF16 weights but failed before generating a candidate:
24,576-token context required **20.64 GiB KV cache**, versus **20.07 GiB**
available under `gpu_memory_utilization=0.85`. Its model capability is unmeasured;
this is an infrastructure stop, not a failed model repair. Weights remain ready.

The **678,478-byte** archive was locally verified and acknowledged at
**22:59:26 UTC**, SHA-256
`1afa3bd3b5d3032775ae150ca43d0df3d24bd8f7f52ed0be2ad30fdff0e523d5`.
Local audit reproduced both reports byte-for-byte, verified all 131 frozen
files, the completed episode grid and exact first-candidate sharing, and
confirmed no 31B completions. SSH subsequently became unreachable. Shutdown
was configured after acknowledgment, but no shutdown-return receipt was captured;
platform power and billing remain unverified.

Separate continuation code lives in `research/diagnostics_resume_v6/`, bound to
the exact stopped archive by `experiments/cuda-diagnostics-v6-resume31b.yaml`.
It runs **only 31B**, with memory utilization **0.90**, retaining all original
tasks, prompts, seeds, BF16, context/output budgets, two concurrent sequences,
evaluation, selection and stopping rules. Its 12-episode / 48-logical-attempt /
40-physical-generation caps are the original per-model allocation. The completed
12B arm and all old source files remain unchanged. The failed startup and memory
change must be disclosed with any later comparison; bitwise invariance across
memory allocations is not assumed.

Local preparation and full metadata validation passed; **19 relevant CPU tests
passed**, including the old iterative runner and the new continuation guards.
This has **not been deployed or GPU-validated**. After GPU restart, deploy the new
namespace and protocol, prepare from the original remote `evidence.tgz`, start
the local `research.diagnostics_resume_v6.monitor` observer, then launch
`research.diagnostics_resume_v6.supervise --prepared <new-preparation>` remotely.
Fresh GPU controls, model hash checks, verified backup acknowledgment and the
existing completion/failure shutdown policy still apply. This continuation does
not include the separate v7 scoped-edit prototype.

Evidence: `evidence/diagnostics-v6/stopped-20261016.json` and
`evidence/diagnostics-v6/resume31b-ready-20261016.json`.

## New GPU endpoint launched; v6 controls passed — 2026-10-16 UTC

The authorized endpoint is now `connect.westd.seetacloud.com:21785`; the logical
SSH alias and local monitor were switched without changing any sealed runtime
source. The instance has an RTX PRO 6000 Blackwell Server Edition with
97,887 MiB reported VRAM, CUDA 13.0 / PyTorch 2.13.0, and a 470 GiB data disk
with approximately 235 GiB free at startup. Both model preparations were retained
and their source/receipt identities verified.

Controller `2026-10-16T222119.856430_0000` started at **22:21:19 UTC**. All six
fresh isolation probes, eight original-fail/fixed-pass checks, and the bounded
API-error-feedback control passed. At **22:23:43 UTC**, the run had entered the
12B first-generation stage and was initializing the model engine; no candidate
had been scored yet. The 31B model loads later and is not yet GPU-validated.

This is the sealed v6 development comparison: 12B/31B, three strategies,
two previously seen defects, two replicates, at most four attempts per episode.
It does not include new training or the local v7 scoped-edit prototype. The
monitor is active on the new endpoint, with report/archive, local hash-verified
backup acknowledgment and shutdown configured after completion or failure.
Evidence: `evidence/diagnostics-v6/gpu-start-21785-20261016.json`.

## 31B weights ready; waiting for GPU — 2026-10-16 UTC

The **62,578,656,403-byte** 31B download and full pinned file-hash/structure
verification completed at **19:12:02 UTC**. CPU staging finished at
**19:17:53 UTC**, sealing both 12B and 31B model identities. At **21:30 UTC**,
live remote validation passed again, the preparation receipt and files were
copied locally and verified, and the local/remote runtime source hashes matched.
The remote instance remains in no-GPU mode (`nvidia-smi` is not executable).
There is no v6 active controller and no new model evaluation result.

The local watcher is running and will launch fresh GPU controls followed by the
v6 iterative repair experiment after a 96 GiB-class GPU becomes available.
GPU loading has not yet been validated. Download and staging are complete;
the status is `ready_for_GPU`. Evidence:
`evidence/diagnostics-v6/weights-ready-20261016.json`.

## Function-scoped repair interface prototype — 2026-10-16 UTC

The operator asked whether Gemma can use execution diagnostics within a bounded
editing scope. A separate local prototype now lives in `research/scoped_repair/`,
with a draft protocol at `experiments/cuda-scoped-repair-v7-draft.yaml`.
It does not alter, queue behind, or launch alongside the active v6 experiment.

Editable functions are enumerated from the supplied before-source, without
consulting human fixes or selecting gold edit locations. All top-level functions
and direct class methods are eligible. Imports, module constants, class
declarations and text outside functions stay byte-identical; the external
`ModelNew.forward` signature is protected. Internal kernel signatures and their
call sites may change together, which is necessary for independent-stride
repairs. Patches apply atomically to a hash-identified current source. The
whole-program comparison interface enforces exactly the same edit boundary.

The implementation parses/compiles syntax without importing or running candidate
code. Editing-contract acceptance does not certify security or numerical
correctness; the existing isolated strict evaluator remains necessary. The
prototype accepts a tool-neutral function mapping. Model output serialization,
prompts, feedback-loop integration and GPU controls are still unimplemented;
there are no new model scores or claims of reliable repair.

**14 new interface checks / 82 total CPU checks passed**, including atomic
kernel/caller edits, protected interfaces, unchanged outside code, stale-source
rejection, no candidate execution, repeated edits and exact round trips for both
public development cases. All **131 v6 sealed files are unchanged**. The local
watcher still reports downloading; no remote process was stopped or modified.

The draft requires matched permitted edits, visible information and attempt
caps across interfaces, with actual cost and failure categories reported.
Failed edits consume attempts; withheld outcomes cannot select or reopen a
repair. Bugs requiring edits outside this scope are out of scope and must be
classified before model evaluation. The existing two defects support development
only; independent cases and a disjoint frozen test remain necessary.

Execution-guided refinement and agent editing interfaces have established prior
work, including [KernelBench](https://arxiv.org/abs/2502.10517) and
[SWE-agent](https://arxiv.org/abs/2405.15793). Combining them is not by itself a
novelty claim. The proposed contribution must be supported by GPU-specific
failure analysis, controlled interventions and measured repair cost.

## Larger Gemma and repeated repair preparation — 2026-10-16 UTC

The operator authorized larger models and repeated revisions. A separate v6
development protocol compares **Gemma 4 12B IT and 31B IT**, both in BF16 without
thinking, using the same two already-seen reduced defects. The 31B official
snapshot is pinned to `842da3794eaa0b77d5f08bae87a17459d91ff475`; every selected
ModelScope mirror file hash matches the official Hugging Face identity. This is
a same-family checkpoint comparison, not an isolated causal parameter-count test.

Each model/task/replicate first produces one candidate shared exactly across
three branches: independent resampling from the original program, revision with
only verdict/count feedback, and revision with detailed visible numerical facts
plus allowlisted compiler/API errors. All branches have at most four candidates
and 8,192 generated tokens per candidate. There are **24 episodes**, at most
**96 logical attempts / 80 unique generations**. First visible strict pass stops
the episode and selects its patch; otherwise the last attempt is selected.
Withheld developer regressions are scored only after all visible-generation
decisions for a model, and never feed back into selection or repair. Actual costs
and shared-first-draw accounting remain explicit. Full-program replacement is
retained; localized patch output would be a separate intervention.

Code: `research/diagnostics_v2/`; protocol:
`experiments/cuda-diagnostics-v6.yaml`. Existing v1.1/v4/v5 source files and scores
remain unchanged. Generation runs in a separate process that must release GPU
memory before isolated candidate evaluation. A fresh six-probe isolation gate,
eight original/fixed checks, and a current-runtime compiler-feedback control must
pass before new model scoring. GPU loading and these controls remain pending.

AutoDL is currently in CPU-only mode (0.5 CPU quota and 2 GiB memory quota).
Cleaning only the reinstallable uv cache, after checking that the installed
environment has no symlinks into it, increased data-disk free space from 59.16 to
73.29 GiB. The installed environment, Python runtime, checkpoints and research
artifacts were retained. The 31B download is approximately 62.58 GB and is running
with resumable, bounded-size range requests and full SHA-256 verification.
CPU staging will seal model identities after download. At **17:13 UTC**, the
downloader and staging worker were both alive, the local systemd watcher was
active, and approximately **3.57 GB / 62.58 GB** had been received (partial bytes,
not certified weights). The watcher resumes interrupted preparation after a
remote container reboot and launches once weights are verified and a 96 GiB-class
GPU is available; this requires the local host/service and SSH to remain
available. Failed experiments are never automatically rerolled. No GPU run has
started.

Final runtime source `bf78dba` passed **68 local CPU tests and 14 remote CPU
tests**; remote source hashes match local files and the preparation seal, and
the frozen original experiment verification passed. The working-directory and
bounded network-retry fixes were tested before restarting preparation, with
earlier receipts retained. Evidence:
`evidence/diagnostics-v6/remote-preparation-20261016.json`.

This is a feasibility study, not evidence that larger Gemma, feedback or multiple
turns improve repair. Independent defects and a frozen formal test set remain
necessary for broader claims. Primary references:
[official 31B model card](https://huggingface.co/google/gemma-4-31B-it) and
[ModelScope distribution](https://modelscope.cn/models/google/gemma-4-31B-it).

The reboot also recovered v5's shutdown-return receipt: command return code **0**
at **14:39:51 UTC**, after verified backup acknowledgment. This confirms command
success; it is not independent platform billing evidence.

## Replay and diagnostic development complete; terminal evidence verified — 2026-10-16 UTC

Both scheduled evaluations finished. The 5,600-program replay ended at
**13:40:25 UTC**; the diagnostic repair grid ended at **14:38:59 UTC**, after
approximately 55 minutes of model generation and scoring. Both archives were
downloaded and hash/size verified. Local terminal audit verified exact attempt
grids, all archived source seals against local files, and byte-identical
regeneration of both JSON and Markdown reports. No new training occurred.

**The strict replay has 0/5,600 successes**, across base and all six C/SFT
checkpoints on 100 previously evaluated tasks. Of the 5,600 results, 4,236 are
contract violations; the rest are crashes, format failures, numerical failures,
or blocked programs. Numerical correctness and custom-compute compliance are
separate endpoints, with unobserved fields retained as unknown. PyTorch compute
was allowed in the original study: these findings do not retroactively establish
cheating or invalidate that study's numerical endpoint. They do prevent claiming
that the original numerical gain demonstrates correct custom CUDA/Triton kernel
generation under this stricter contract.

The diagnostic development grid completed **84/84 attempts**, with no unknown
outcomes. Base resolved **2/4** attempts with grouped diagnostic facts and **0/4**
in each of the flat-facts and verdict-only conditions. Both successes are two
samples of the **same independent-strides defect** (`development/case-002`),
passing visible and withheld developer regressions. All C policies combined
resolved **0/36**, and all SFT policies combined resolved **0/36**. Flat and grouped
diagnostic facts were verified identical. Failure counts: 49 numerical failures,
19 crashes, 11 format failures, and 3 contract violations; two of the format
failures reached the output cap.

This is a small descriptive signal about diagnostic presentation, not evidence
of general diagnostic gains, an intrinsic model limit, or statistically reliable
fine-tuning harm. Two already-seen reduced defects from one repository cannot
support those claims. Review the successful stride repairs and the failure
categories before expanding to independent development defects and a separately
frozen formal test set; do not resume training merely to seek an RL win.

The diagnostic archive was locally acknowledged at **14:39:46 UTC**. SSH became
unreachable afterward and is still unreachable at the 16:31 UTC observation.
The supervisor is configured to request shutdown after acknowledgment, but no
shutdown-return receipt was captured; platform power/billing status remains
unverified. Both local monitor services exited, and local status files now
record completion with verified backups rather than a stale backup-wait stage.

Evidence: `evidence/diagnostics-v5/completed-20261016.json`.
Audited reports: `runs/diagnostics-v5/terminal-audit-20261016/diagnostics/REPORT.md`
and `runs/diagnostics-v5/terminal-audit-20261016/replay/REPORT.md`.

## Diagnostic evidence is the new primary question — 2026-10-16 UTC

The operator selected: **does Gemma fail to repair GPU code because useful
execution evidence is missing, or because it cannot use available evidence?**
Existing RL/SFT checkpoints are comparison policies; no new RL training is
scheduled. This supersedes the old single-turn-only scope for the authorized
repair follow-up, without changing the original study or its scores.

`experiments/cuda-diagnostics-v5.yaml` and `research/diagnostics_v1/` implement a
bounded development experiment: two previously seen reduced upstream defects,
seven saved policies, three feedback conditions, two generations per cell (84
attempts). Each episode starts with identical broken code and one repair attempt.
The conditions are verdict-only, flat measured facts, and the same facts grouped
by test/input. The two richer conditions have identical fact hashes; the flat
condition is canonical diagnostic JSON, **not** arbitrary unfiltered stderr.
Measurements include visible input layouts and numerical mismatch witnesses.
They contain no root-cause guess or upstream fix. PR locators are removed from
comments with executable AST identity checked and license notices retained.

Controls require original-fail/fixed-pass on visible and withheld developer
regressions, plus a fresh six-probe isolation gate. Withheld scores never enter
prompts or choose a completion. Format/truncation failures remain in the grid;
infrastructure failures stop execution. All policies receive the same output
budget; actual prompt tokens, generated tokens, and evaluation costs are retained.
The context preflight rejects overflow instead of truncating one feedback arm.

All 54 selected local CPU tests passed; original frozen-source verification
passed. Current CPU preparation is in `runs/diagnostics-v5/prepared-20261016-c/`. No new
repair result or GPU-control PASS is claimed here. The supervisor can wait for
the active v4 replay, require its verified backup, then run controls and repairs,
report, archive, obtain local backup acknowledgment, and request shutdown. The
new namespace leaves the active v4 source seal unchanged.

At **05:13:25 UTC**, the successor was actually queued on AutoDL (PID **56646**):
`runs/diagnostics-v5/controllers/2026-10-16T051325.446441_0000`.
All **14 remote CPU checks passed**, and the downloaded receipt/JUnit, prepared
manifest, and complete runtime source hashes match locally. Both monitoring
services are active. The predecessor was at **2,353/5,600** replayed programs;
the successor is in `waiting_for_previous_verified_backup`, with no diagnostic
GPU work started. It defers predecessor shutdown through the existing handoff
marker; source files and scoring remain unchanged. The new monitor takes over
global status only after the verified handoff. Evidence:
`evidence/diagnostics-v5/remote-queue-20261016.json`; live diagnostic status:
`runs/DIAGNOSTICS_RUN.json`.

The initial cross-machine validation correctly stopped on two local-only
post-hoc statistics files absent from AutoDL. They are not diagnostic runtime
dependencies. The v5 seal now explicitly excludes those two files; all actual
runtime dependencies remain bound. A fresh `prepared-20261016-b` preserves the
superseded preparation and avoids adding files to v4's active source inventory.
Before any model generation, tokenizer measurement found the longer original
program needs 5,284 output tokens, exceeding the initial 4,096 cap. Preparation
`c` raises the common cap to 8,192 and context to 24,576, adds a source-output
budget guard, and preserves `b`. This is a pre-generation feasibility correction,
not an outcome-driven rerun or per-condition budget change.

Two reduced defects from one repository are integration/development evidence,
not a confirmatory test set. A failure with these diagnostics does not establish
that the information was sufficient or that Gemma is intrinsically incapable.
Next research gate: establish usable diagnostic signal/cost, then curate unseen
real defects and independently review diagnostic relevance before freezing the
formal study. Counterexample minimization, strong-model comparison, and complete
repository repairs are later work, not implemented findings.

CPU preparation command (use a fresh output directory):
`python -m research.diagnostics_v1.prepare --out runs/diagnostics-v5/prepared-NEXT --policies runs/reliability-v2/transfer-v3/prepared-20261013-b/policies.json`

## Interface development complete; retrospective replay continues — 2026-10-16 UTC

The completed 84-attempt interface grid was downloaded before the longer replay
finishes. Archive size/hash, all source identities, the exact 7 × 6 × 2 attempt
grid, and JSON/Markdown report regeneration were verified locally. This preserves
the finished development evidence independently of the eventual terminal backup.

| Arm | Full program strict successes | Fixed host strict successes |
|---|---:|---:|
| base | 0/6 | 2/6 |
| C | 0/18 | 6/18 |
| SFT | 1/18 | 8/18 |

These are six development tasks, one draw per saved policy/condition, with three
saved seeds for C/SFT and one shared base. Repeated policies are not independent
new tasks. The counts provide a descriptive interface-support signal; they do
not establish an RL advantage, new-test transfer, or a general causal mechanism.
Full-program scoring includes format/ABI failures before runtime, so source
and failure-category review is required before interpreting the contrast. No
program was repaired or rerolled after observing these outcomes.

At the 01:18 UTC check, decomposition remained active at 1,247/5,600 scored
attempts. This is a partial known-task replay, not a completed endpoint. The
workstation watcher is active; terminal report/verified backup/shutdown remain
enabled. The approximate remaining time from the elapsed average is 21 hours,
with substantial variation in per-task compilation and evaluation cost.

Evidence: `evidence/reliability-v2/capability-v4-interface-complete-20261016.json`.
Local snapshot: `runs/reliability-v2/capability-recovery/20261015T185011.215170Z/interface-snapshot/`.
Archive SHA-256: `0629028fadae86b51c878f53995630a91919557c951a8c878f5b77628e3a3209`.

## Current-source GPU controls passed; interface sampling started — 2026-10-15 18:52 UTC

The replacement controller passed **6/6 isolation probes** and **14/14 GPU
controls** with the repaired completion extractor. Controls include real Triton
and CUDA execution, numerical-only PyTorch, wrong math, dummy-kernel/PyTorch
fallback, initialization/forward crashes, initialization-vs-forward separation,
and all six fixed hosts over four dtype/layout cases each.

At **18:52:56 UTC** it entered the 84-attempt interface development phase;
base model loading/generation was observed at the next check. Zero attempts
had final scores at that observation. This is an actual running evaluation
pipeline, not a new training run or a claimed model improvement. Its controller
is `runs/reliability-v2/capability-controllers/20261015T185011.215170Z`.
The 5,600-program retrospective decomposition is scheduled immediately afterward;
it has not started at this observation. Automatic report, verified backup and
terminal shutdown remain enabled.

## Fixed-host extraction repaired; replacement controller running — 2026-10-15 UTC

The first v4 controller was paused before any policy generation because review
found that the reused v1 extractor requires a `ModelNew` class. That would reject
valid kernel-only responses in the new fixed-host condition. The old v3 prompts
requested complete programs, so this issue does not revise their reported scores.
A separately versioned extractor now shares identical code-fence/thought handling
between the two v4 conditions, followed by the correct condition-specific AST
contract. No output is repaired, retried or silently stripped of host code.
The GPU controls now traverse extraction as well as assembly and execution.

The session lost terminal access again while that controller was SIGSTOP-paused.
At **18:47 UTC**, restored access found PID 1916 stopped and GPU utilization/memory
both zero; its child had completed all 14 controls. After preserving receipts,
logs and the exact prior source in an archive, the paused parent was retired.
No interface attempts or old-program scores had been produced. Archive SHA-256:
`a722b141b6c8bfa32fcfb671b52b30f88f42d10546d0626e6f1b62dc678ae628`
(**90,618 bytes**), locally verified at **18:49:37 UTC** under
`runs/reliability-v2/capability-recovery/20261015T170848.349323Z/`.

Updated validation: **39 CPU checks passed, 1 optional Torch check skipped**;
all **8** capability checks passed on AutoDL. Deployment hashes match, and the
original v1.1 source/configuration seal still verifies. New controller PID
**4989** started at **18:50:11 UTC**:
`runs/reliability-v2/capability-controllers/20261015T185011.215170Z`.
At launch it was repeating current-source isolation/GPU controls; no new model
result is claimed by this entry. The unchanged schedule is 84 development
interface attempts, followed by all 5,600 saved primary-policy Level-2 attempts.

Automatic reporting, verified-backup acknowledgment and terminal shutdown are
active in the replacement controller. The workstation watcher was restarted
with restart-on-failure; it now distinguishes a stopped process from a running
controller. Prior shutdown defer markers belong only to the retired controller.
Live observations remain in `runs/reliability-v2/capability-recovery/latest.json`
and `runs/ACTIVE_RUN.json`.

## Capability decomposition and fixed-host development launched — 2026-10-15 UTC

The workstation terminal and SSH access recovered. The operator booted AutoDL;
96 GiB-class RTX PRO 6000 and approximately 60 GiB free data-disk space were
observed. No preceding project workload was found. The previous October 13
controller's shutdown return receipt was recovered: return code 0 after the
verified-backup acknowledgment (command success, not platform billing evidence).

Separate follow-up code is under `research/reliability_v2/`; protocol:
`experiments/cuda-capability-v4.yaml`. Original v1.1 source/configuration still
verify. Local checks: **38 passed, 1 optional Torch check skipped**; all **7** new
CPU checks also passed on AutoDL. All ten deployment artifact hashes matched.
No new training or confirmatory test-task generation is authorized by this runner.

- Same-program decomposition selects **all 5,600 saved Level-2 attempts** from
  shared base, all three final C seeds and all three final SFT seeds (100 tasks,
  eight draws). Original program bytes, reference identities, model/input seeds,
  four checks and dtype domain are retained. Initialization, forward completion,
  numerical correctness and custom-forward compliance are recorded separately.
  Profiling adds an observed forward call and removes timing measurement; this
  is a separately versioned retrospective replay. Missing observations are null,
  never relabeled as demonstrated numerical or custom-kernel failure. Previously
  permitted PyTorch computation is not retroactively called cheating.
- The interface development run schedules **84 attempts**: the same seven policies,
  six existing development graphs, two conditions, one draw. Both conditions use
  Triton with an identical named ABI, grid and output specification. One requests
  the entire program; the other appends a tested allocation/metadata/launch-only
  host wrapper. Both have the same 4,096-token generation cap; actual prompt and
  generated tokens are recorded. The wrapper contains no mathematical solution.
  This is a development floor/cost check, not the formal interaction experiment.

At **17:08:48 UTC**, remote controller PID **1916** launched at
`runs/reliability-v2/capability-controllers/20261015T170848.349323Z`.
At the initial observation it was running fresh isolation controls; policy scoring
had not started. It must pass all six isolation probes and 14 new evaluator/host
controls before proceeding to interface sampling and then saved-program replay.
Any harness error stops the run and preserves partial evidence. Completion or
failure writes reports/archive, waits for a SHA-256-verified local backup, and
requests the previously authorized shutdown. The local watcher is
`cuda-gemma4-capability-v4.service`; it now attempts to collect the shutdown
return receipt after acknowledging backup instead of immediately exiting.

Live local status: `runs/reliability-v2/capability-recovery/latest.json` and
`runs/ACTIVE_RUN.json`. No new scientific result is claimed at launch. Formal
new-task manifest/reference review, interaction uncertainty analysis, and actual
model repair trials remain later work; two original/fixed repair controls alone
are not model repair results.

## Supplementary p-values: conditional on saved policies — 2026-10-14 UTC

At the operator's request, a **post-hoc** analysis adds two-sided, null-centered
studentized bootstrap tests for the three original Level-2 correctness contrasts.
All ten saved policy grids were read and validated; original sources, scores,
protocol and percentile intervals remain unchanged. Average each policy's eight
draws within task, average saved seeds within arm, then resample **100 paired task
vectors**. Training seeds and generated programs are not treated as extra tasks.
The null is zero mean task difference conditional on these saved checkpoints.

With 100,000 resamples and RNG seed 20261014:

| Contrast | Difference (percentage points) | Raw bootstrap p | Bonferroni p |
|---|---:|---:|---:|
| C minus base | +17.1250 | 0.000010* | 0.000030* |
| S minus base | +16.7917 | 0.000010* | 0.000030* |
| C minus SFT | +0.7500 | 0.518535 | 1.000000 |

\* Zero of 100,000 bootstrap draws exceeded the observed statistic. These values
are the plus-one Monte Carlo reporting floor, not exact tail probabilities or
proof of a zero p-value. Correction covers the three existing contrasts, not
every historical exploratory analysis. The original percentile intervals and
these studentized tests are different procedures, not exact inverses.

Validity is approximate and assumes independent, representative task vectors
with nonzero finite variance. KernelBench tasks are curated, previously observed
and may share related structure; this analysis does not establish that sampling
assumption. It does not estimate population uncertainty over training seeds,
prove RL/SFT equivalence, establish unseen-task generalization, or identify a
mechanism. Existing task-bootstrap intervals have the same task-sampling scope.
No normality or exact treatment-label exchangeability is claimed. Degenerate
all-equal task samples return no inferential p-value rather than claiming equality.

Implementation: `research/reliability_v2/task_inference.py`; **7 CPU tests passed**,
including null-centering, sign/scale invariance and degenerate-sample handling.
Run: `.venv/bin/python -m research.reliability_v2.task_inference --out runs/reliability-v2/task-inference-20261014 --repeats 100000 --seed 20261014`
(use a new output directory to rerun). Full output:
`runs/reliability-v2/task-inference-20261014/analysis.json`; committed receipt:
`evidence/reliability-v2/task-inference-20261014.json`. This was local CPU analysis;
no new model generation, training or remote GPU work occurred.

## Strict development pilot complete: no passing policy sample — 2026-10-13 UTC

The 36-attempt development pilot completed at **20:20:23 UTC**, taking
**695.27 seconds**. Recomputing the archived report locally reproduces both
JSON and Markdown exactly; the complete unique policy/prompt/task grid and
metrics hash match. Results below combine the two prompt conditions across
six development graphs, one draw each, using only the first training seed:

| Policy | Attempts | Strict correct | Numerically correct |
|---|---:|---:|---:|
| base | 12 | 0 | 4 |
| C-20261021 | 12 | 0 | 3 |
| SFT-20261021 | 12 | 0 | 1 |

There were **24 compilation/runtime crashes, 9 contract violations, 2 format
failures and 1 incorrect result**. Crash logs include invented API arguments,
integer pointers passed to Triton, bad extension linkage and unsupported views
of non-contiguous inputs. Observed contract failures include PyTorch arithmetic
or dtype conversion. No row is labeled a harness error; the observer source
matches the passed controls and archived run. This is diagnostic evidence, not
a proof that every possible evaluator defect has been excluded.

These results provide **no positive evidence of strict custom-kernel transfer**.
The small development sample cannot establish a population effect or a
three-seed comparison. Earlier permissive-contract correctness gains remain
valid for that contract; they do not establish improved custom CUDA generation.
The two successful upstream-fix reproductions remain evaluator controls, not
model repair successes. Formal new-task generation and model repair evaluation
have not run. Before expanding evaluation, resolve the development floor with
equal executable interface scaffolding and budget-matched visible-feedback
repair trials, retaining this first attempt and the unchanged strict contract.

The **325,674-byte** archive was downloaded, SHA-256 verified and acknowledged
at **20:21:09 UTC**; its digest is
`e6c164dde41f80839d280ac1ab76d3956fbef9f936f130b9a78e0907b2dfbd70`.
The controller is configured to shut down after this acknowledgment. A fresh
SSH connection timed out, and no shutdown-return receipt was obtained;
platform power/billing state is therefore unverified. Evidence:
`evidence/reliability-v2/transfer-development-complete-20261013.json`.
Report:
`runs/reliability-v2/transfer-development-recovery/20261013T200847.197395Z/extracted/runs/reliability-v2/transfer-controllers/20261013T200847.197395Z/pilot/REPORT.md`.
Original v1.1 source/configuration verification still passes.

## CUDA transfer and repair follow-up started — 2026-10-13 UTC

The operator explicitly retained CUDA and authorized strict new-task evaluation,
real-defect repair, and experiments explaining transfer, with **an official
Kaggle award / Google event selection** as the intended outcome. Public release
alone is not the success criterion. Positive effects and award competitiveness
remain unestablished. This extends the original single-turn scope for evaluation;
there is no new training and the original v1.1 experiment remains immutable.

Protocol draft: `experiments/cuda-transfer-v3-draft.yaml`. Primary policies remain
shared base plus all three final C and SFT seeds; S is secondary, retaining its
documented early stop. Primary endpoints are strict correctness on new graphs
and hidden-test repair success at three attempts. Six endpoint/comparison pairs
share one familywise error budget. Identical API scaffolding and budget-matched
feedback/no-feedback interventions test behavioral explanations; they cannot
establish an internal neural mechanism or full causal mediation by themselves.

Implemented in the separate `research/reliability_v2/` namespace:

- Runtime contract observation: per-input Torch dispatch operations and actual
  CUDA profiler events, with positive calibration before candidate execution.
  All **7 GPU controls passed**, including genuine Triton and CUDA C++ positives,
  unused-kernel and dummy-kernel-plus-PyTorch negatives. This bounded observation
  model does not prove resistance to arbitrary native in-process corruption.
- Trusted profiler initialization occurs before the unchanged seccomp filter.
  All **6 inherited C1 hostile/recovery probes passed** for this launcher. The
  first attempted profiler initialization was blocked; that evidence is retained.
- **48 distinct expression candidates** in six families plus **6 separate
  development graphs**; **216 CPU finite/domain checks passed**. These checks do
  not independently prove mathematical references or justify a test-set freeze.
- Real-fix provenance capture pins upstream PR, first parent, fix revision,
  source files and license hashes. Three upstream fixes are archived. Two Liger
  bugs (softcap tail masking and independent token/mask strides) now reproduce:
  original versions fail numerical checks in both regression sets; fixed versions
  pass correctness and the strict contract. These are **reduced development
  reproductions**, not full-repository repairs or model repair achievements, and
  are excluded from the future test set.
- Repair episode logic separates visible feedback from final hidden scoring,
  fixes the submission selection rule, caps generation/test budgets and records
  actual token usage. CPU verification: **24 passed, 1 existing optional Torch
  check skipped** in the CPU-only venv. The original source seal still passes.

Evidence receipt: `evidence/reliability-v2/transfer-development-controls-20261013.json`.
Prepared policy/prompt identities: `runs/reliability-v2/transfer-v3/prepared-20261013-b/`.
The preparer verifies all ten saved policy identities, including one shared base;
it explicitly reports that confirmatory evaluation is not ready.

At **20:08:47 UTC**, controller PID **5546** restarted a bounded **36-attempt
development pilot**: six development graphs, two prompt conditions, one draw,
and the predeclared base/C-20261021/SFT-20261021 policies. The first check found
base model loading/generation in progress. This first-seed pilot measures
integration and cost only; it must never replace the planned three-seed result.
Controller: `runs/reliability-v2/transfer-controllers/20261013T200847.197395Z` on
AutoDL. Local watcher: `cuda-gemma4-transfer-development.service`. Completion or
failure produces a diagnostic report and archive; the watcher verifies the
download before acknowledging it, followed by the authorized terminal shutdown.

The first controller (`20261013T200456.397879Z`) stopped before generating or
scoring any program: `FileNotFoundError: [Errno 2] No such file or directory:
'ninja'`. The operational entry had omitted the original `scripts/gpu_env.sh`.
Its report/archive were downloaded and hash-verified (SHA-256
`9b90230beefa82fbea08221a968534c9fbb63572053cd30def79454fc9502d73`). The restart
uses that existing launcher and checks compiler availability before loading
weights. The old controller's shutdown was explicitly deferred to protect the
new run; no model result was discarded.

Remaining before confirmatory runs: review references/domains and overlap with
prior tasks; curate the independent real-repair test set (target 24 defects from
three repositories); finish analysis and end-to-end budget checks; estimate
development cost and interval precision; then bind and freeze the separate
protocol. No test-candidate outputs have been requested. No publication or
external outreach has been performed.

Resume read-only status with:
`systemctl --user status cuda-gemma4-transfer-development.service` and
`runs/reliability-v2/transfer-development-recovery/latest.json`.

## Measurement replay complete and backed up — 2026-10-13 UTC

The sealed 80-program replay finished at **03:35:22 UTC**, after about five
hours. All 12 evaluator controls passed. Independent local verification
matched the archived scores/report and evaluator code to their hashes, and
all program identities, original records and input seeds to the manifest.
Receipt: `evidence/reliability-v2/replay-complete-20261013.json`.

All **21 previously audited ENOSPC failures passed numerical correctness**
on replay: C-20261023 recovered seven, S-20261023 six, and SFT-20261023 eight.
The other 59 programs retained their original result components. Across this
one-task replay, correct results increased from 40/80 to 61/80; this selected
task total is not a whole-benchmark performance estimate. Shared base stayed
at 2/8. This is a measurement correction using saved programs, not new model
learning, a new-task generalization result, or proof of custom-kernel execution.
Original v1.1 scores remain immutable. Its primary Level-2 comparisons are
unaffected; an additional RL benefit over SFT and broad speed gains remain
unestablished. The changed GPU driver still precludes attributing old/new
timing differences solely to the evaluator repair.

The terminal report and evidence were downloaded, SHA-256 verified and
acknowledged at **03:36:02 UTC**. Replay archive SHA-256:
`de83d2378cb98e59a7ff4a64552d0b9e37c62c243772769e9f379d77bc022f2e`.
Local evidence/report directory:
`runs/reliability-v2/remote-recovery-20261012/20261012T222455.458848Z/`.
The earlier 30,779,004,695-byte whole-study archive also finished downloading
and passed SHA-256 verification at **2026-10-12 22:40:07 UTC**; its receipt is
`runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/verified-archive.json`.

The controller requested authorized shutdown at **03:36:37 UTC**, with backup
acknowledged, and recorded return code 0. A fresh read at **18:36:49 UTC**
found the container had started about 83 seconds earlier: SSH was reachable,
the old controller PID was absent and no GPU compute process was listed.
Thus the completed run is not active on the newly started container. This
observation does not establish the intervening platform billing/power state.
The detailed live snapshot and shutdown receipts are saved in
`terminal-live-status.json` within the local evidence directory above.

## All streaming controls PASS; 80-program replay running — 2026-10-12 22:37 UTC

The requested restart detected the restored GPU and launched controller PID
1445 at `runs/reliability-v2/controllers/20261012T222455.458848Z`. Its explicit
cleanup receipt confirms 14,763,975,122 bytes of preceding C3 tensor scratch
were hashed and removed, restoring free disk from 49,524,518,912 to
64,288,628,736 bytes. Sources, logs, verdicts, timings, policies and original
scores were preserved. New C0–C3 gates passed; the new C3 tensor scratch was
also hashed/cleaned before the large controls. The prior failed controller's
shutdown command has now been confirmed with return code 0.

All **12 streaming-evaluator controls passed** at 22:37:21 UTC, including the
full-size reference and handwritten Triton L1Norm controls, each with four
successful numerical checks. This fixes the previously observed disk-space
block without changing the evaluator, its limits, or the sealed 80-program
manifest. The PASS receipt is byte-identical to the remote record:
`evidence/reliability-v2/streaming-controls-20261012T222455.458848Z.json`.
Local checked metadata and full-size control verdicts are under
`runs/reliability-v2/restart-20261012/verified-checks/`.
Handwritten control performance is not a trained-policy result, and these
controls do not certify the future strict runtime implementation contract.

The controller entered **replay** at 22:37:22 UTC. At the startup check, the
first program was in progress and zero of 80 final replay rows were complete;
do not treat launch as completed evaluation. This run includes shared base
and all nine saved policies, using unchanged program sources and input seeds.
The workstation watcher now reads the actual nested `controls/cases/checks.json`
location, as well as replay progress. Report/archive/hash-verified backup and
authorized terminal shutdown remain scheduled. The old whole-study archive
continues independently with its stale shutdown disabled for this follow-up.

## Replay stopped before policy evaluation: storage guard; restart prepared — 2026-10-12 UTC

The controller `20261012T204658.744430Z` passed fresh C0, C1, C2 and C3 on the
GPU boot. All ten small v2 controls passed: reference/Triton execution,
shape/dtype rejection, unchanged-NaN handling, actual mutation detection and
sandbox read/write/network boundaries. Full-size L1Norm reference preparation
then stopped with `Insufficient evaluator storage: free=15165579264,
required=17181107920`. No policy replay began, and the full-size custom control
was not reached. The new space check caught this before a candidate was scored.

The failure report and evidence were downloaded and SHA-256 verified locally:
`runs/reliability-v2/remote-recovery-20261012/20261012T204658.744430Z/`.
Archive SHA-256:
`1aaac9815cc99753aec7651d33ce15a30589705d40e9a40eec2a77adde3840b4`.
The four passing boot-gate records were imported from this verified archive.
At 22:18 UTC SSH timed out, after the controller's configured failure/shutdown
grace period. A shutdown return receipt has not yet been retrieved; do not
equate SSH unreachability alone with verified platform power state.

Operational restart preparation retains the same evaluator and 80-program
manifest. C3 calibration leaves large reproducible `.pt` input copies and
`.safetensors` output files. The new operations helper selects only those
files under an explicitly passing C3 gate, hashes them before deletion, and
retains source, logs, verdicts, timing samples and a cleanup receipt. Its CPU
self-test checks confinement, protected files, failed-gate rejection and
idempotency. The controller will clean the preceding C3 scratch, rerun boot
gates, clean only the new passing C3 tensor scratch, then verify at least
48 GiB plus metadata space before full-size controls. Actual reclaimed bytes
will be recorded on the node; no storage allowance is weakened.

## Follow-up GPU controller launched — 2026-10-12 20:46 UTC

Deployment fast-forwarded the remote checkout to `6b92dc2`; all transferred
files were hash-checked, the original source seal passed, and all 14 audit/
storage CPU tests passed in the remote torch environment. Controller PID 1704
started at `runs/reliability-v2/controllers/20261012T204658.744430Z` and entered
C0 revalidation. It runs C0–C3, commits each gate PASS, then the sealed v2
controls and all-policy replay. A failed prerequisite prevents later stages;
there are no new generation or training requests in this controller.

Remote operational entry point:
`runs/deploy/reliability-v2-20261012/run_reliability_replay_20261012.py`.
The workstation service `cuda-gemma4-reliability-replay.service` observes the
controller and downloads/hash-verifies its terminal evidence before sending
the backup acknowledgment. The controller creates a terminal report/archive
on success or failure, allows up to one hour for backup, and invokes the
already-authorized shutdown. A failed control has a 15-minute inspection
grace period before shutdown. Actual completion and backup require receipts;
the running C0 observation is not a control/replay PASS.

Latest machine-readable observations:
`runs/reliability-v2/remote-recovery-20261012/latest.json`.
The prior large archive continues separately, with its stale shutdown guarded.

## GPU follow-up prepared: versioned storage replay — 2026-10-12 UTC

The operator attached a GPU for the follow-up. The node reports RTX PRO 6000
Blackwell Server Edition, 97,887 MiB VRAM, 25 CPU cores of quota and 120 GiB
RAM. Driver 595.58.03 differs from the formal confirmation node's 580.95.05;
rerun C0–C3 and record the environment before replay. The data disk has about
60 GiB free, insufficient for the old L1Norm tensor-copy/output peak.

The new `research/reliability_v2/` evaluator stages each generated input once,
moves it into the sandbox, and removes that case file after loading it. It
retains all four outputs for parent-owned comparisons. Storage preflight
accounts for scratch and headroom. Shape/dtype are checked before writing,
and unchanged NaNs no longer look like output mutation (nonfinite outputs
still fail numerical correctness). Isolation/limits are copied from C1, with
only owned staging-file handling and repository-root resolution changed.
The original executable source and results are unchanged.

`runs/reliability-v2/replay-20261012/manifest.json` seals all 80 saved attempts
on `level1/38_L1Norm_`, including four format failures. Shared base and all
nine saved policies are included, using their original four input checks and
seeds. Manifest SHA-256:
`dd7b52e6b273eea6d1d7cbbfc3a844cfe369ef71dc6c66599238b693dc49a246`.
Code hashes are bound to it; a code change requires a new manifest/control run.
The CPU suite passes (13 tests, one torch-dependent skip in the small venv;
all seven storage/persistence tests pass in the existing torch environment).

GPU controls and replay are still pending at this preparation record. Controls
cover ordinary and Triton execution, wrong output metadata, unchanged NaNs,
real input/output mutation, sandbox boundary checks, and full-size reference
and custom-kernel cases. Only matching successful controls unlock replay.
This version retains the original permissive implementation contract; the
strict runtime-attribution/new-task experiment is not yet implemented.
The changed driver and evaluator are disclosed alongside old/replayed scores;
do not treat a timing difference as an isolated effect of the storage repair.

The old terminal-backup service was stopped, amended and restarted resumably.
Its completion now defers acknowledgment/shutdown while the new follow-up is
active. This avoids the old study powering off a new workload. The follow-up
retains the operator's report, backup and shutdown-on-completion/failure policy.

## Reliability follow-up: CPU audit complete, new evaluation still a draft — 2026-10-12 UTC

The immediate objective is to explain the correctness gain and test whether it
persists when actual custom-kernel execution is required. The v1.1 study remains
frozen. New analysis lives in `research/reliability_v2/` because adding Python
files to its old executable directories would invalidate the original source
seal. `experiments/reliability-v2-draft.yaml` is an exploratory follow-up plan,
not a preregistration or permission to start an unvalidated GPU evaluation.

The read-only audit validated and parsed all 9,600 final programs. It imports or
executes none of the candidates. Reproduce with:

```bash
.venv/bin/python -m research.reliability_v2.audit --out runs/reliability-v2/audit-NEW-ID
```

The current generated report and full provenance are under
`runs/reliability-v2/audit-20261012-r2/`. Seven focused audit tests pass, and
`tools.replication verify` still validates the original source/configuration.
The earlier `audit-20261012/` output is retained; r2 separates storage I/O hints
from compiler diagnostics instead of matching every occurrence of `error:`.

Descriptively, API/import/name traceback hints occur in 188 of the base's 960
programs, versus a mean of 16.33 per 960 for C, 23.67 for S, and 31.67 for SFT.
These are log categories, not demonstrated causal mechanisms or uniformly
hallucinated APIs. Exact `forward` AST matches are 94 for base and mean 92 for
C: this narrow copy indicator did not increase for C, but cannot rule out
equivalent reference reuse, different wrappers or PyTorch fallback. A fixed
96-case diversity review queue is generated; detailed review is pending and
will not be used as an unbiased prevalence estimate.

The audit also found **21 output-serialization ENOSPC failures**, all on
`level1/38_L1Norm_`: eight SFT-20261023, seven C-20261023 and six S-20261023.
The archived tracebacks end in the harness's `safetensors.save_file`; the old
evaluator recorded them as `crash` before numerical comparisons. Historical
filesystem exhaustion versus quota is not established by the logs. The task's
large tensors and both private/sandbox input copies require explicit storage
accounting. Original results stay intact; these failures are unresolved
measurement cases, not established model mistakes. These 21 cases do not enter
the primary Level-2 comparison. Other storage hints include six Level-2
`File too large` failures; source inspection shows added convolution padding
relative to the references in those cases, so oversized candidate outputs
must be distinguished from insufficient storage for valid outputs. They must
not automatically be declared infrastructure failures or successful programs.

The next steps are:

1. Finish the source/log review and validate a separately versioned evaluator
   with free-space accounting, output shape/dtype checks, and a NaN-safe
   mutation detector. Re-evaluate all saved programs for the ENOSPC task across
   base and all nine policies, with original inputs/seeds and side-by-side
   original/replay reporting; do not cherry-pick only failed trained programs.
2. Validate runtime implementation controls: legitimate custom kernels pass;
   reference replay, unused kernel definitions, and dummy kernels followed by
   PyTorch computation fail only the new custom-kernel contract. Permissive
   v1.1 results retain their original meaning.
3. Seal new tasks, prompts, sampling/input seeds and analysis before new test
   outputs. The draft proposes 48 new operator-graph tasks across six families,
   eight programs per task/condition, all saved policies, and both minimal and
   explicit API-scaffold prompts. A separate bridge on known tasks measures
   contract sensitivity. Task sources, controls and feasibility remain pending.
4. Evaluate these fixed checkpoints, then write the mechanism/generalization
   story supported by the results. Additional training or a second model is
   deferred until this diagnosis; RL need not outperform SFT for the study to
   report an informative result.

All nine inference policies and final result grids are now backed up and
verified locally; the last S receipt is dated 17:44:53 UTC. The larger terminal
archive still transfers through the resumable backup service and is not yet
verified. No new GPU experiment has been launched.

## R3 complete; final report retrieved; terminal backup recovery — 2026-10-12 UTC

All nine formal jobs and their 960-program final evaluations are complete.
R3 passed at 11:04:51 UTC; this gate certifies completion and valid evidence,
not a positive RL-versus-SFT finding. The controller recorded `complete` with
no error, generated its terminal report at 11:05 UTC, and archived the study
before its authorized shutdown. The final job, `S-20261023`, has 721/960
correct (75.1042%), including 128/160 on Level 1 and 593/800 on Level 2.
Its full score/generation grids, checksums and step-40 adapter/config match
were validated remotely against the unchanged frozen source.

| Policy | Overall correctness | Primary Level-2 correctness |
|---|---:|---:|
| Base, shared independent evaluation | 55.9375% | 54.8750% |
| C, mean of three training seeds | 73.4722% | 72.0000% |
| S, mean of three training seeds | 72.9167% | 71.6667% |
| SFT, mean of three training seeds | 72.1528% | 71.2500% |

The preregistered Level-2 task-cluster comparisons, averaged over the saved
training seeds and using 98.333% family-adjusted intervals, are:

| Contrast | Correctness change, percentage points [interval] |
|---|---:|
| C minus base | +17.1250 [12.6250, 21.7917] |
| S minus base | +16.7917 [12.3333, 21.4167] |
| C minus SFT | +0.7500 [-1.9583, 3.5833] |

The evidence supports improved GPU-program correctness after fine-tuning on
this known task set. It does not establish an additional C/RL advantage over
SFT, equivalence between them, broad runtime acceleration, or a gain isolated
to handwritten CUDA. The intervals condition on these saved policies and
resample tasks; they do not quantify population uncertainty over training
seeds. The first S seed stopped at step 15 and remains included at its actual
checkpoint, with the documented NaN detector confound and unequal exposure.

Both generated reports, their JSON companions, the R3 gate, controller outcome,
shutdown receipt and final S request/completion metadata were retrieved and
SHA-256 checked locally (17 files) under
`runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/extracted/`.
The confirmation Markdown report hash is
`5392f9005876b0c3d113f47dd20f9ab675dfdeea367273667b7de3dcc853eaf4`.
The R3 evidence commit `dc5dc20` was imported without changing executable code.
The final S evidence archive passed local transfer, full hash and 7,808-file
manifest validation at 17:21 UTC. All ten final grids (one shared base plus
nine trained policies, 9,600 programs) were then revalidated from local copies;
all three primary Level-2 comparisons reproduced the remote report exactly.
The receipt is `terminal-recovery-20261012/local-analysis-verification.json`
under the same backup root. This is local verification of existing evidence,
not an additional independent experiment.

`shutdown-return.json` records exit code 0. At the 17:17 UTC check, SSH was
reachable in a newly started container, the former workers were absent, and
`nvidia-smi` was unavailable with permission denied. Thus the historical
shutdown succeeded at the command level; the machine is currently reachable
again, and no new GPU experiment is running.

The full 30,779,004,695-byte archive remains on the remote data disk with SHA-256
`8cc3de5ecadc11e6ce5339dacba4bafc702e0f79d4892f57f927cb07b9f01e74`.
Its first workstation copy was interrupted; neither retained partial copy was
verified or acknowledged. The old non-resumable watcher was stopped and replaced
by `cuda-gemma4-terminal-backup.service`, which first retrieves and validates the
last S evidence/policy, then resumes the full archive, checks its full hash,
acknowledges it, and invokes the already-authorized shutdown. The eight earlier
jobs already have verified local result and inference-policy backups; the
latest C backup completed at 03:57 UTC. All nine jobs' final results are now
verified locally. The last S inference policy subsequently passed local
verification at 17:44:53 UTC; full local archival remains pending.
Recovery records and future verification receipts are under
`runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/`; this recovery
changes only operational scripts under `runs/deploy/`, never frozen source.

## Eight R3 jobs evaluated; last seed trained and generating final programs — 2026-10-12 UTC

Eight of nine formal jobs have complete final evaluations. `C-20261023`
completed 40 steps and all 960 final programs at 21:28 UTC on October 11:
733 correct (76.3542%), with 134/160 on Level 1 and 599/800 on Level 2.
The complete score/generation grids, request/score/generation checksums,
final adapter/config match and unchanged frozen source were validated.
The per-seed summary is `runs/deploy/completed-c23-20261012/summary.json`.

This third C seed exceeds the corresponding SFT result by 4.4792 percentage
points overall. The descriptive primary Level-2 C-minus-SFT contrast is +3.125
points, with a 98.333% task-cluster interval of [-1.25, 7.375] points. Each of
the three single-seed Level-2 C-minus-SFT intervals crosses zero, and the
second seed has the opposite direction. These observations do not establish
a stable RL-specific benefit; the final all-arm report remains pending.
They also do not establish equivalence. The third C seed's common-correct speed
ratios are 1.0462 on Level 1 and 1.0002 on Level 2, without evidence of broad
kernel acceleration.

The last job, `S-20261023`, has saved step 40 and completed all three
120-program intermediate evaluations. At 03:29 UTC on October 12 it had
generated 720/960 final programs; scoring had not started. Its final adapter
and configuration match the evaluation request, and the partial generation
grid passed validation. The original supervisor/worker PIDs remain 1693/1706;
the GPU and workstation monitor are active, OOM counters remain zero, and
approximately 90 GiB of data-disk space is free. Comparable complete final
evaluations suggest about 8–10 hours remaining for experiments and reporting
if execution continues normally; bulk terminal backup time is additional.
The authorized terminal report, archive and shutdown workflow remains intact.

The third SFT seed's evidence and inference policy passed local verification
at 19:12 UTC on October 11. All seven earlier jobs now have verified local
results and inference policies, and its completed transfer service is disabled.
The latest C evidence and inference policy are archived remotely under
`runs/deploy/completed-c23-20261012/`; their workstation copy is managed by
`cuda-gemma4-c23-backup.service`. Evidence SHA-256:
`cc2730dd385dcc380284399bda86b259470146ce130042d7bd348027a9a5a2e5`;
policy SHA-256:
`02c4dfa6a6ca4ca4461ff3f06718c95adb2eed0c04638610be7753ce8cd37454`.
Consult receipts under
`runs/autodl-westb-reboot-20261006/completed-c23-20261012/` before claiming the
newest local backup complete. Optimizer states remain remote for terminal
archival; these interim backups do not acknowledge the terminal study archive.

## Seven R3 jobs evaluated; third C seed in final evaluation — 2026-10-11 UTC

Seven of nine formal jobs have complete final evaluations. `SFT-20261023`
completed 40 steps and all 960 final programs at 07:38 UTC on October 11,
with 690 correct (71.875%): 116/160 on Level 1 and 574/800 on Level 2.
The complete score/generation grids, request/score/generation checksums,
final adapter/config match and unchanged frozen source were validated.
Its descriptive correctness gain over the shared base grid is 15.9375
percentage points. All three SFT seeds are now complete, with overall
correctness of 71.25%, 73.3333% and 71.875%. These results continue to support
a benefit from fine-tuning; they do not establish an additional RL benefit.
The all-arm confirmatory report remains pending the last two jobs. Conditional
common-correct speed ratios for this SFT seed are 1.0248 on Level 1 and 0.9882
on Level 2, without evidence of broad kernel acceleration.

The eighth job, `C-20261023`, has completed 40 training steps and its three
120-program intermediate evaluations. All 960 final programs were generated;
216 had been scored at the 19:09 UTC check. The partial score grid and complete
generation grid passed validation. `S-20261023` is the only unstarted job.
The original supervisor/worker PIDs remain 1693/1706, the GPU is active,
OOM counters remain zero, and approximately 92 GiB of data-disk space is free.
The workstation monitor is active. Comparable completed jobs suggest roughly
another day for the remaining experiments, assuming no new interruption;
terminal backup time is additional. The authorized terminal report, archive
and shutdown workflow remains in place.

The sixth job's evidence and inference policy passed local backup verification
at 20:39 UTC on October 10; its completed transfer service has been disabled.
All six earlier jobs now have verified local results and inference policies.
The newest SFT result and policy have been archived remotely under
`runs/deploy/completed-sft23-20261011/` and their workstation transfer is
managed by `cuda-gemma4-sft23-backup.service`. Evidence SHA-256:
`2bf6648042aa62e940ac9a63095b0de56c247ae64915123d7f992ef484673a54`;
policy SHA-256:
`45e71224c72ff47f676886619c3254f1e539e8c053b8a92aecd31fb2002238fb`.
Check the receipts under
`runs/autodl-westb-reboot-20261006/completed-sft23-20261011/` before claiming
this newest local backup complete. Optimizer states remain remote for terminal
archival. These interim backups do not acknowledge the terminal study archive.

## R2 complete; six R3 jobs evaluated; third SFT seed underway — 2026-10-10 UTC

Six of nine formal jobs have complete final evaluations. The latest,
`C-20261022`, completed 40 steps with no stop and all 960 final programs at
18:27 UTC on October 10: 684 correct (71.25%), comprising 122/160 on Level 1
and 562/800 on Level 2. The complete held-out grid, request/score/generation
hashes, adapter match and unchanged frozen source were validated. The per-seed
summary is `runs/deploy/completed-c22-20261010/summary.json`.

The second C seed is 2.0833 percentage points below the corresponding SFT seed
overall. On primary Level 2, C-minus-SFT is -2.375 points, with a 98.333%
task-cluster interval of [-6.875, 2.25] points. This descriptive single-seed
contrast and the first seed's +1.5-point Level-2 contrast both cross zero;
their directions differ. Fine-tuning continues to show positive correctness
changes relative to base, but the available comparisons do not demonstrate
a stable RL-specific advantage over SFT. No formal three-seed R3 aggregate or
PASS is claimed. Common-correct speed ratios for the latest C seed are 1.0174
on Level 1 and 1.0013 on Level 2; these do not establish broad acceleration.

The seventh job, `SFT-20261023`, has scored all 320 TRAIN candidates and saved
step 10 of 40. At 20:29 UTC, its intermediate evaluation had scored 75/120
programs and was advancing. The original supervisor/worker PIDs remain
1693/1706; no new OOM or study stop was observed, and approximately 94 GiB of
data-disk space was free. The persistent workstation monitor is active.
The remaining sequence is this SFT job, C-20261023, then S-20261023. Based on
completed job durations, the remaining experiments are expected to take about
1.5–2 days if execution continues normally; terminal backup time is additional.

The new C evidence and inference policy are archived remotely under
`runs/deploy/completed-c22-20261010/`. Evidence SHA-256:
`b0d9f85b22622532985d1958e318eab5954132d9206ac9f39b8137ca6e351cde`;
policy SHA-256:
`2a152b34f99e5a858b2f0c08686e0f42de7f73f0504aa0aef119a116530635aa`.
The resumable workstation transfer is managed by
`cuda-gemma4-c22-backup.service`; consult the verification receipts under
`runs/autodl-westb-reboot-20261006/completed-c22-20261010/` before claiming
this newest backup complete. All five earlier completed jobs have verified
local result and inference-policy backups. Optimizer states remain remote
for terminal archival.

### Earlier October 10 progress

Five of nine formal jobs have complete final evaluations. The latest,
`SFT-20261022`, completed 40 steps and all 960 final programs at 04:44 UTC on
October 10, with 704 correct (73.3333%). It had no stop and improves 17.3958
percentage points over base. The same-seed S result is 708/960 (73.75%);
both have 581/800 correct on Level 2. These similar descriptive rates do not
establish equivalence or an RL-specific benefit. The complete held-out grid,
request/score/generation hashes, adapter match and frozen source were checked.

Earlier, `SFT-20261021` completed 40 steps and 960 programs at 01:47 UTC on
October 9, with 684 correct (71.25%); `S-20261022` completed 40 steps and 960
programs at 15:29 UTC on October 9, with 708
correct (73.75%). Neither new job hit a stop criterion. The latter is a second
S training seed with a positive descriptive change versus base (55.9375%).
Every completed grid, request/score/generation hash and final adapter/request
match was checked; the frozen source verified unchanged.

The first matched-seed C result is 72.8125%, only 1.5625 percentage points above
SFT overall. On primary Level 2 the descriptive C-minus-SFT contrast is 1.5
points; its 98.333% task-cluster interval is [-3.625, 6.5] points. This uses the
frozen bootstrap settings but is an interim single-seed contrast, not the
preregistered all-three-seed aggregate. The interval crosses zero. SFT itself
improves 15.3125 points over base overall; improvement over base therefore does
not by itself establish an RL-specific advantage. Full training-seed results
and controls are still required. No formal R3 aggregate or PASS is claimed.

The sixth job, `C-20261022`, finished its 120-program step-10 evaluation at
06:15 UTC on October 10 and reached step 20, with checkpoint evaluation underway
by 06:41 UTC. The original remote supervisor and worker retained PIDs 1693/1706
through another local monitoring outage. The same host key and GPU UUID were
verified; no new OOM or study stop was observed. The workstation's temporary
SSH files and old monitor process were absent. Monitoring now uses a persistent
host-key pin, a restricted project-specific SSH key, and an enabled user systemd
service that restarts after failure and starts on user login. The watcher has a
fixed October 15 observation deadline and does not equate that deadline with
study completion. Live status remains in `runs/ACTIVE_RUN.json`.

The SFT-20261022 evidence and inference policy were archived under
`runs/deploy/completed-jobs-20261010/`. Their workstation copy completed and
passed verification at 08:07 UTC on October 10; the completed transfer service
has been disabled. Evidence SHA-256:
`1d288ff8f6acd092614b6e149faa764a76cc63a5a1cb19f6c4a8e2a9856a6928`;
policy SHA-256:
`82e7a39d19a3245fc59d14999a9242e4a6298d1490b899214c7927ffb076de3f`.
Verification receipts are under
`runs/autodl-westb-reboot-20261006/completed-jobs-20261010/`. Both archive
hashes, all manifest files, the complete 960-program grid and final adapter
match passed validation after successful transfers.

New complete evidence and the two inference policies were archived separately
under `runs/deploy/completed-jobs-20261009/`. Evidence archive SHA-256:
`cf38f652be48ad2a77f32aeaa9f2e5131c6ec65ffd0637772b4c3decf78da92f`;
policy archive SHA-256:
`f6d440bbf4969bd2e65bd963409bd2a13cc4ea52b822eee92eb792e0aeca21cf`.
A detached local backup completed at 17:23 UTC: both transfer exit codes were
zero; archive hashes, all 14,283 manifest files, both full 960-program grids and
their adapter/request matches passed validation. Its receipts are under
`runs/autodl-westb-reboot-20261006/completed-jobs-20261009/`.
The two inference policies and complete evidence are now verified on the
workstation; optimizer states remain remote for terminal archival.

### Prior milestones and detector audit — 2026-10-08 UTC

Independent reevaluation completed all 2,880 programs on the frozen 120-task
grid. R2 passed at 10:28 UTC and its evidence was committed:
`evidence/gates/r2-20261007T102812.799328Z-78f0cfe4.json`.
The generated comparison report is
`runs/replication-v1.1/reevaluation/REPORT.md` and its JSON companion.
Both C and S retain positive correctness changes over base; the preregistered
Level-2 task-cluster intervals exclude zero after the family adjustment.
These intervals condition on the original trained checkpoints and do not
establish reproducibility across training seeds. Common-correct speed ratios
remain close to one; this is not evidence of broad kernel acceleration.

TRAIN-only SFT learning-rate selection finished all three candidates and their
validation grids, selecting `1e-5`; its extra cost remains separate from the
formal matched candidate budgets. The first of nine formal R3 jobs,
`C-20261021`, finished all 40 training steps and its complete 960-program final
evaluation at 01:56 UTC. Its step-10/20/30 evaluation grids also completed.
The validated per-seed summary is retained under
`runs/deploy/C-20261021-complete-20261008/seed-summary.json`; it records another
positive correctness change versus the shared base grid. This is a single new
training-seed result, not the completed three-seed or RL-versus-SFT comparison.
No subset of seeds is promoted to the formal R3 aggregate.

The second job, `S-20261021`, completed its step-10 evaluation and then triggered
the declared `detector_finding` stop at optimizer step 15. One of its 120 TRAIN
programs (sampling step 14, call 5, index 1) received `output_persistence`; it was
also rejected as `nonfinite` and earned reward 0. There were no harness errors
or sandbox violations in the stopping group. This is not established reward
exploitation. Its actual step-15 final evaluation completed all 960 programs at
12:36 UTC: 671 correct (69.8958%), compared with 537/960 (55.9375%) for the shared
base grid, a descriptive gain of 13.9583 percentage points. Level 1 had 125/160
correct and Level 2 had 546/800. The frozen requests, complete grids and
request/generation/score checksums were validated. This is a stopped-run result,
not a completed 40-step S run or evidence that C is superior to S.

Two of nine formal jobs now have complete final evaluations. At 18:51 UTC the
third job, `SFT-20261021`, had saved step 40 of 40 with no stop criterion. Its
TRAIN candidate pool contains all 320 programs; step-10, step-20 and step-30
held-out evaluations each completed 120 programs. All 960 final programs were
generated and 32 had been scored. The complete intermediate grids, final
generation/partial score grids, step-40 adapter/request hashes and frozen source
were validated. SFT's final comparison and the other two training seeds remain pending.
Do not promote these partial jobs to the confirmatory three-seed aggregate.

Source review identified a detector confound: the persistence check uses
`torch.equal`, which rejects unchanged NaNs. A trusted CPU-only probe on the
same remote PyTorch version confirmed that a separately allocated, bytewise
unchanged NaN output triggers this predicate while a finite control does not.
The flagged Triton source also contains incorrect reductions and a row mask
that leaves output elements unwritten. Raw output tensors were already cleaned
by the evaluator, so the exact cause of this particular persistence flag is
unconfirmed without an instrumented replay. No generated candidate was executed
for this diagnosis, no GPU replay was run alongside scoring, and the frozen
detector, original verdicts and stop remain unchanged. The shorter S training
exposure must be reported; it cannot be described as a matched 40-step run.
The probe and stop evidence are under
`runs/deploy/S-20261021-detector-audit-20261008/`.
The remote step-15 policy/evidence archive is 255,285,460 bytes, SHA-256
`4915d84199a448cb32b0371fbdd4d757a32d0c49c9c2e6579be763310f07d99c`.
The workstation copy was verified at 16:55 UTC: its archive SHA-256, all 1,153
manifest files, the step-15 adapter/request match, complete step-10 grid and
61-row partial step-15 snapshot passed validation. This earlier snapshot is
separate from the subsequent full 960-program result. Optimizer checkpoints
remain in the remote study for terminal archival.

The complete S evaluation was separately copied to
`runs/autodl-westb-reboot-20261006/S-20261021-complete/` and verified at 16:57 UTC.
Its 9,866,851-byte archive has SHA-256
`ed6ed5d88c4ca3d159f82591297ffb0a5edea7aa8117fd62a55a8a724d362548`.
All 5,102 manifest files, the full 960-program grid and its match to the already
backed-up step-15 adapter passed validation. The retained per-seed summary is
`runs/deploy/S-20261021-complete-20261008/seed-summary.json`.

The original remote supervisor and workstation backup watcher remain active.
The controller retains the frozen sequence, terminal report, archival and
authorized shutdown; live progress is in `runs/ACTIVE_RUN.json`.

## Active continuation — westb v1.1 after reboot, 2026-10-06 UTC

Reconnected after the operator reported a reboot. The original hostname, GPU
UUID and driver are unchanged; a trusted CUDA matrix-multiplication smoke
passed. The old controller had no outcome, report, archive or shutdown record
and no worker remained. Its interruption cause is not established. A post-hoc
interruption report is explicitly marked as such and preserves the prior logs.

Validated the complete base evaluation (960 programs), C's complete generation
grid (960 programs) and all 66 finished C scores, including their declared seeds,
request hashes and the base completion marker. S has not started. No rows were
discarded or reclassified. Copied the study's metadata, programs, scores and logs
to the workstation and verified archive and individual score/generation hashes:
`runs/autodl-westb-reboot-20261006/pre-resume-evidence.tgz`, SHA-256
`4b053de08cb7f0146c0b6cb9f60f7d473cf4de7daa13408a1971adf81b5dbd33`.

Removed only abandoned, regenerable tensor/work files from the interrupted C
attempt and the previously invalidated ownership audit, after the verified
backup. The retained scratch manifest accounts for 81,603,348,284 bytes; free
space after cleanup was 129,150,660,608 bytes. Original programs, logs, scores,
checkpoints and interrupted attempt directories remain. Recovery records are
under `runs/replication-v1.1/recovery/20261006-reboot/`.

Current controller: `runs/replication-v1.1/controllers/20261006T211103.982319Z`.
It reuses the passed R1 audit on the same hardware and calls the unchanged
`Study.reevaluate()` followed by `Study.confirm()`. C resumes at sample index
66 in a new attempt directory; completed base/C scores are reused. The frozen
source, seeds, resource limits and scientific settings are unchanged. R2's
declared decision still governs whether the nine formal C/S/SFT runs may start.
The operational supervisor retains the progress/wall guards, PASS-evidence
commits, terminal report, archive and authorized shutdown. The new workstation
watcher follows this exact controller, rather than the interrupted one.

Resume command used (already running; do not launch a duplicate):
`scripts/gpu_env.sh .venv/bin/python -u runs/deploy/resume_westb_r2_r3_20261006.py --power-off`.
Current process IDs are in `runs/ACTIVE_RUN.json`; current watcher status is
`runs/autodl-westb-reboot-20261006/latest-status.json`.

### Original westb deployment

The operator provided the westb SSH endpoint on port 15577 and reiterated:
finish the run, write the report, then shut down. Revision v1.1 was merged into
the preserved remote history and its source seal verified. All 92 CPU tests and
a real-tokenizer contract check passed on this host. Hardware: RTX PRO 6000
Blackwell (97,887 MiB), 25-CPU / 120-GiB quota, driver 595.58.03, nvcc 13.0.88;
250G data disk with 139G available at startup.

R0 and fresh C0–C3 passed on the new physical GPU. The GPU KL probe passed with
disabled-adapter/base log-probability difference 0.0 and finite replay values.
This does not establish the cause of the historical step-35 KL spike. R1 passed
all controls, GRPO/SFT restart checks and the SFT reevaluation at 09:03 UTC:
`evidence/gates/r1-20261006T090346.470792Z-5a83857b.json`. R2 subsequently started;
R3 has not started. The older R1 FAIL in gate history belongs to v1.

Interrupted controller: `runs/replication-v1.1/controllers/20261006T074520.201312Z`.
Passed audit: `runs/replication-v1.1/audit/20261006T074521.673972Z`.
The controller writes a terminal report before archiving and invokes the AutoDL
shutdown command after completion/blockage. The detached operational wrapper
commits each new PASS record. The workstation watcher follows this host and
filters its status to the current boot/protocol, retaining the old failed
protocol separately. Current PIDs/endpoint are in `runs/ACTIVE_RUN.json`;
workstation status is `runs/autodl-westb-20261006/latest-status.json`.

Raw new-host calibration JSON and logs were copied into the study's
`node-evidence/20261006T073417.471534Z/`, including the raw C2 timing samples,
so terminal archival retains them alongside the gate records.

The first westb audit was explicitly interrupted before an R1 decision because
tar extraction restored the workstation UID 1000 and mode 0700 on the study
root. The sandbox UID could not traverse this ancestor and failed during Torch
bootstrap with `No usable temporary directory found`, before candidate source
execution. The retained attempt has `invalidated-deployment.json`; its raw rows
and controller outcome were not overwritten. Restored the study root to remote
root ownership and the normal traverse-only ancestor mode, then verified a
trusted CUDA/Triton sandbox smoke in that same directory. The frozen source,
scientific configuration and containment policy are unchanged. Restarted R1
in a new audit/controller directory; its KL probe and first reference controls
passed. The interrupted attempt is not a completed R1 pass or a policy result.
Raw C2 samples additionally have an archive-safe copy outside any `work/` path.

## Local repair checkpoint — before westb deployment

The cloned host passed R0 and C0–C3, then R1 failed before training with
`TypeError: unsupported operand type(s) for +: 'BatchEncoding' and 'list'`.
Transformers 5.17.0 defaults `apply_chat_template` to `return_dict=True`.
The shared prompt-token helper now explicitly requests `return_dict=False` in
both the KL replay and SFT encoder. All 92 CPU tests pass. A separate CPU check
with the actual saved Gemma tokenizer and pinned Transformers reproduced the
old return type, checked all eight retained TRAIN replay completions, verified
identical prompt tokens and correct SFT label masks, and passed. This is not a
GPU R1 PASS. Evidence: `runs/deploy/tokenizer-compatibility-20261006.json`.

The operator additionally requested a report before shutdown. The supervisor
now writes `TERMINAL_REPORT.md` and JSON before its terminal archive. Successful
confirmation links the complete statistical report; a failed audit is labelled
incomplete and preserves its error rather than claiming a research result.
`experiments/replication-v1.1.yaml` retains every scientific setting from v1;
only the run namespace, compatibility fix and reporting code change. The old
protocol and its failure evidence remain intact. The repaired entry point is
`bash scripts/start_replication.sh all`, using `runs/replication-v1.1` after its
fresh R0 validation. Local R0 passed at
`r0-20261006T072220.933953Z-c3487e6e.json`; source revision at freeze was `160408d`.
The remote must independently revalidate R0. This revision has not been deployed
to the powered-off node.

The previous controller requested automatic shutdown under the operator's
earlier blockage rule. Its 74,269-byte archive was copied and SHA-256 verified
locally (`56e4f0e7721d3599379936fba20f4e3455bafd0c5dcbbf5b566850246af39fc4`),
then SSH became unreachable. No shutdown return record was observed. Restored
evidence and the post-hoc local report are under
`runs/recovered-clone-20261006/runs/replication-v1/`.
R2 and R3 did not run; no new policy-improvement result exists from this attempt.
Restart requires the operator to power on the GPU instance, then source/protocol
deployment, fresh R0 and the R1 retry. No remote process is claimed active.

## Cloned GPU node — 2026-10-06 UTC

The operator supplied a cloned AutoDL node on SSH port 31578. The clone has the
same RTX PRO 6000 Blackwell class (97,887 MiB), a 22-CPU / 110-GiB cgroup allocation
and a 250G data disk with 153G available at startup. Its driver is 580.95.05,
different from the pilot node; nvcc remains 13.0.88. Source revision `cbe5b9a`
and the sealed replication protocol verified unchanged. The clone independently
passed R0 at `r0-20261006T071001.859221Z-1f74ad2b.json`.

Detached operational wrapper
`runs/deploy/clone_bootstrap_20261006.py` re-runs C0–C3 on this physical GPU
before handing off to `scripts/start_replication.sh all`. It commits each new
PASS record and archives evidence before authorized shutdown if bootstrap fails.
The existing replication controller handles its own progress watchdog and
terminal archive/shutdown. Bootstrap directory:
`runs/clone-bootstrap/20261006T070955.347946Z`.

The workstation watcher `runs/deploy/watch_clone_20261006.py` monitors the new
SSH endpoint for up to eight days, discovers the replication controller, and
starts checksum-verified archive retrieval when an archive is ready. Its status
is under `runs/autodl-clone-20261006/`; `runs/ACTIVE_RUN.json` records the current
PIDs and endpoint. The watcher uses an authenticated SSH master, without saving
the password. The bootstrap subsequently passed C0–C3 before the R1 failure
described above. Historical gate PASS records alone do not certify another host.

## Phase 2 — authorized implementation and prospective protocol

Preparation completed on 2026-10-06 UTC (2026-10-05 Vancouver time). Local and
AutoDL CPU suites each passed 88 tests. R0 passed independently on both machines;
the latest record is `runs/gates/r0-20261006T064334.343992Z-e789f325.json`.
The sealed job list contains all nine arm/seed combinations and the three
960-program pilot re-evaluation requests. R1–R3 have not run; GPU compatibility
and new scientific outcomes remain unverified. The no-card allocation has a
2 GiB / 0.5 CPU quota, so preparation never loaded the model.

The operator approved independent re-evaluation, multiple training seeds and a
verified-success SFT control, then explicitly requested implementation and no-GPU
AutoDL preparation. Unlimited AutoDL budget and shutdown on completion/blockage
remain authorized. Generated programs still execute only in the C1 sandbox;
the request to disable the assistant's broken local terminal sandbox does not
remove that experimental containment requirement.

`experiments/replication-v1.yaml` fixes three NEW training seeds for C, S and SFT,
eight independent generations per task and four fresh correctness checks for
reported evaluations. The old 120-task split is unchanged and already observed;
this is a replication on known tasks, not an unseen-task confirmation. The old
pilot is retained separately, including C's actual step-35 KL stop.

The four new executable gates are R0 (CPU artifact/protocol validation), R1
(GPU reference-invariance, reference-replay control, three-step GRPO and SFT
save/restart/re-evaluation), R2 (complete independent base/C/S re-evaluation),
and R3 (all nine trained final policies and their comparison report). R0 does
not certify SFT GPU compatibility. R1–R3 must actually run on the GPU before
their gates can pass. Both pilot gains disappearing in re-evaluation blocks
mass training for diagnosis; a negative result is retained, not discarded.

SFT generation budget is 320 base-model TRAIN candidates per seed. Selection
uses one fresh check, matching the original GRPO training verifier, with no
detector flags and normal generation termination. SFT receives the original
completion token IDs and masks all prompt tokens from the loss. No deduplication
changes the empirical success frequency. A separate TRAIN-only tuning seed
compares three fixed learning rates using 32 fresh validation programs each;
ties choose the smaller rate. Formal runs share that choice. Tuning overhead,
tokens and verification calls are separate costs, not a claim of identical FLOPs.

Primary follow-up comparisons are C–base, S–base and C–SFT on Level 2. Their
task-cluster intervals use a Bonferroni family adjustment. Level 1, fast1/fast1.1/
fast2, common-correct speed ratios and source/failure taxonomy remain visible.
Intervals are conditional on the saved training seeds; seed-specific results
are reported separately. Missing/duplicate rows, infrastructure failures,
altered score hashes or missing training seeds cannot enter a complete report.
Static source markers do not establish the executed backend or causality.

Workflow: freeze with `python -m tools.replication freeze`; pass R0 with the
verified pilot archive; after attaching the GPU, run
`bash scripts/start_replication.sh all` in a detached session. The supervisor
applies a 30-minute progress watchdog, preserves results/checkpoints and invokes
the AutoDL shutdown script at completion/blockage. It refuses a no-card boot and
refuses to shut down the local development host. `tools.watch_pilot` can monitor
the new directory under `runs/replication-v1/controllers/` for a hash-verified
workstation backup.

Local pilot backup: the full archive's SHA-256 was independently recomputed and
matches `d58e5119237fe80eeb3553d3d9fe800f30ce75f6ad6d0509bba7d28bff1e96fc`.
Metadata was recovered under `runs/recovered-pilot/`; the final C/S adapters were
also extracted from that verified archive. The remote completion record says
`complete` and `/bin/bash /usr/bin/shutdown` returned 0. No new GPU work is
claimed by this preparation entry.

Further mechanism experiments (including the conservative-prompt intervention)
remain a separate prospective protocol after the replication audit. Current code
implements descriptive source/failure decomposition and the reference control;
it does not label those as causal evidence or a completed new research finding.

## Latest autonomous pilot outcome

Status: **complete**. Detail: `None`.
Run: `runs/autonomous/20261005T035943.378043Z`. Checkpoint/evidence preservation and the authorized platform shutdown follow this record.

## Training continuation authorized (2026-10-05 UTC)

The operator said “开始训练”; combined with the earlier unlimited AutoDL budget,
this authorizes C4–C6 and the preregistered pilots after their prerequisite gates.
It supersedes the old stop-before-C4 instruction. C1–C3, the sandbox-escape stop
and the fixed held-out split remain mandatory.
C0–C5 passed. C4 selects Gemma 4 12B on the existing LayerNorm TRAIN task after
the documented curriculum narrowing below. C5 completed three effective GRPO
steps, verified the reload hash and GPU reclamation, and re-evaluated the saved
LoRA policy on fresh inputs. Its metadata is backed up locally under
`runs/c5/20261005T034343.616845Z`; full checkpoints remain on the AutoDL data disk
and are included in the terminal autonomous backup.

C6 is prepared for two frozen pilots, C and S, each up to 40 steps with evaluation
every 10 steps and at early termination. Each arm keeps the kickoff's 8-hour
ceiling, reserving a final-evaluation window. The fixed full held-out grid remains
uninspected. `tools/autonomous_pilot.py` supervises progress, archives checkpoints
and evidence, then invokes `/bin/bash /usr/bin/shutdown` on completion/blockage.
The local watcher verifies the archive hash and acknowledges a completed copy
when reachable; the remote data-disk archive survives power-off either way.

The later autonomous-decision authorization is used for one explicit curriculum
continuation if that retry fails: LayerNorm alone has measured mixed outcomes,
whereas GELU hits invalid model-generated Triton APIs. The initial two-task count
was an implementation choice, not kickoff's P_signal threshold. Validate the
narrower existing TRAIN subset before training; retain every prior gate failure
and record the operator authorization in the new gate. No reward or held-out
split change is authorized by this decision.

The operator subsequently authorized autonomous decisions and requested power-off
on completion or blockage. Preserve checkpoints/evidence and use AutoDL's
`/usr/bin/shutdown` (the platform command) after backup; do not release the instance.

AutoDL supports Landlock ABI 1 and dropped UIDs, but `unshare -Urnm true` returns
`Operation not permitted`; cgroups are read-only. Resource containment therefore
uses rlimits plus a parent watchdog (50 ms target polling; not a hard cgroup RSS
or VRAM reservation). Workers are serialized, use UID 64532, and get their own
workdir/cache. `/root` traversal is changed from 0700 to 0701 for that UID to reach
the interpreter, with Landlock denying access to unrelated home/project files.
The trusted CUDA bootstrap precedes thread-synchronized seccomp; generated code
never runs until the network filter is active. See LEDGER for the compatibility
failures and the Landlock v1 cache adaptation.

Required reward-design reading was skimmed before implementing scoring:
[KernelBench](https://arxiv.org/html/2502.10517v1),
[Kevin](https://arxiv.org/html/2507.11948v1),
[CUDA-L1](https://arxiv.org/html/2507.14111v1),
[Sakana's corrected report](https://sakana.ai/ai-cuda-engineer/), and
[METR](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/).
Design implications: isolate reference outputs from candidate allocators; use
fresh inputs and persistence checks; separate compilation from measurement;
interleave reference/candidate timing and retain raw noise. KernelBench fast0
is correctness and fast1 requires speedup > 1; report these separately.

## Historical GPU bring-up — C0 passed (2026-10-04 UTC)

The operator attached the GPU and requested continuation. C0 passed on its first
formal attempt, with all five checks passing and the wrapper exiting 0. Source
revision was `b9eca731ce7a9442b6966f914cc826e3db7fab70`, with a clean working tree.
The immutable gate record is committed at
[`evidence/gates/c0-20261004T224213.417987Z-7b8a8dac.json`](evidence/gates/c0-20261004T224213.417987Z-7b8a8dac.json),
SHA-256 `c99d4bdc7414a473c93e4f6b187c5b22055bb515a488dc09b5866e3aa1742398`.

- Hardware: RTX PRO 6000 Blackwell Server Edition, 97887 MiB VRAM; cgroup quota
  25 CPUs / 120 GiB RAM; 250G data disk with 183G available after C0.
- Driver 595.58.03 advertises CUDA 13.2. Installed nvcc is 13.0.88; the actual
  CUDA reference worker reports torch 2.13.0+cu130 / CUDA runtime build 13.0.
- The 12B snapshot loaded all 677 tensors plus `Gemma4UnifiedConfig` and
  `GemmaTokenizer`; E4B loaded all 2130 tensors plus `Gemma4Config` and
  `GemmaTokenizer`. Both local chat templates worked. The earlier no-GPU metadata
  probe limitation is resolved for these snapshots on the GPU allocation.
- vLLM 0.30.0 loaded 12B in bfloat16 and returned exactly `READY` (normal stop).
  It selected TRITON_ATTN for Gemma 4's heterogeneous attention heads. The model
  used the pinned sampling preset and text-only multimodal limits.
- Both pinned training-split matrix-multiplication references executed on CUDA
  and produced finite outputs. This proves reference plumbing, not candidate
  correctness, speedup or sandbox isolation. All 35 preparation tests also passed.
- After the workers exited, NVML reported 0 MiB used and 0% GPU utilization.
  No inference server or training job was left running.

Raw evidence on the node:

- `runs/setup/c0-detached-20261004T223830.330028Z.log` and
  `runs/setup/c0-exit-20261004T223830.330028Z.txt`.
- `runs/c0/20261004T223947.649004Z/`: separate JSON/log pairs for each weight
  loader, real generation and CUDA reference worker.
- `runs/gates/c0-20261004T224213.417987Z-7b8a8dac.json`.
- `runs/preflight/node-readiness-20261004T223757.034769Z.json`.

All raw records are backed up locally under `runs/autodl-c0/20261004/runs/`.
The gate JSON is also mirrored into local `runs/gates/` for the scripted status
table below; its committed evidence copy is byte-identical.

**C0 PASS; C1–C6 not run.** Timing noise floor, detector coverage, base-model
pass@1/pass@8, P_signal and rollouts/hour remain unmeasured. No generated kernel
candidates or hostile programs were executed. Ledger observations: transient
startup utilization and a first-inference Triton JIT; neither is a gate failure.
No hypothesis, task split or sampling preset changed.

Next development step is C1 isolation, followed by C2 timing and C3 cheat checks.
Those gates remain disabled until their implementations exist. Reproduce C0 only
when needed with `cd /root/autodl-tmp/cuda-gemma4-llm && bash scripts/run_autodl.sh c0`.
GPU attachment, storage, model access and budget are no longer open questions.

## Reuse and conflicts — recorded before implementation (2026-10-04)

Read `kickoff.md` in full, then `/mnt/fastdata/glomergence/LEDGER.md` and
`REPORT_STATUS.md`. Reuse source HEAD: `e602f63207fe4e67eeda4dff7b53d49e92cc318b`.
This repository initially contained only `kickoff.md` and was not a Git repository.

| Source | Planned reuse | Conflict / adaptation |
|---|---|---|
| `scripts/setup_node.sh` | uv-managed Python 3.12; validated GPU pins; AutoDL mirrors | Remove Java/Maelstrom and hf-mirror fallback; use explicit ModelScope weight directories; do not alter shell profiles. |
| `scripts/gpu_env.sh` | Conda isolation, bfloat16-era stack environment, allocator setting | CUDA C++ kernels need a real matching compiler; do not mistake CUDA driver capability for installed toolkit. |
| `tools/gate_g4.py` | Gemma sampling 1.0 / 0.95 / 64; text-only multimodal limits | No Qwen presets or old task distribution. |
| `env/reward.py` | Reasoning-channel stripping and Python extraction | Reject unfinished reasoning; no Maelstrom statuses or reward function. Reward design waits for the required reading and C1–C3. |
| `train/launch.py`, configs | bfloat16, separate re-eval memory budget, kill-criterion lessons | Maelstrom callbacks cannot evaluate CUDA while vLLM sleeps. Keep launcher disabled until the new evaluator and C4 exist. |
| `eval/report.py` and held-out-grid pattern | Reproducible reports, CIs, frozen split | Bootstrap unit must reflect task/program dependence. Do not import old distributed-systems metrics or results. |
| CPU sandbox | Lessons only | GPU access, compilation, memory reclamation and infinite-kernel recovery require a new C1 design. |

Kickoff §6 requests both a gate JSON in each passing commit and no `runs/` in Git.
Keep raw gate records under ignored `runs/gates/`; copy passing evidence to tracked
`evidence/gates/` and record its SHA-256. No gate pass is claimed for preparation.

## Inventory (initial observations; scripted record below is authoritative)

- Local GPU: RTX 5070 Ti, 16303 MiB total, 561 MiB used at inspection;
  driver 595.71.05 advertises CUDA 13.2. Still smoke tests only.
- 16 logical CPUs, about 30 GiB RAM, about 605 GiB free on `/mnt/fastdata`.
- Default Python is Conda 3.13.9; its nvcc is also Conda-provided. No `/usr/local/cuda*` found.
- Previous project's uv Python 3.12.13 environment has torch 2.13.0, Triton 3.7.1,
  vLLM 0.30.0, transformers 5.17.0, TRL 1.14.0 and PEFT 0.21.0.
- At initial local preparation, no AutoDL node or spending authorization was supplied; superseded by the remote inspection below.

## Session outcome (2026-10-04)

Operator steering: **local preparation only**. Completed the scaffold and initial
import smoke, separate uv development environment, pinned task loader and split,
ModelScope downloader/verification, C0 worker and gate orchestration, local
preflight/reporting, tests, and an AutoDL runbook in `README.md`.

`scripts/setup_node.sh` and `scripts/gpu_env.sh` are adapted copies. Gemma sampling
and the extractor are adapted from the old project. The GPU reference smoke uses
the old interpreter read-only, with no package changes there. Training, reward
scoring, the sandbox, timing, cheat detectors and research evaluation remain
explicit stubs; their full reuse/adaptation follows prerequisites, not this session.

No model weights were downloaded, no rental node was provisioned, and no training
or generated candidate execution occurred. Formal C0–C6 gates have not run.

## Ledger entries added this session

- Tool sandbox startup failure blocked the original document read; recovered after environment change.
- Old setup used hf-mirror and insufficient download validation; adapted to verified ModelScope snapshots.
- KernelBench default shapes can be large; restrict local smoke to inspected official training references.
- Reconcile gate-evidence commits with ignored `runs/` using byte-identical evidence copies.
- A combined delete/add patch was rejected; use a single update operation for existing files.
- Resolving the Python symlink bypassed the venv; preserve it and retain the failed preflight record.
- Staged whitespace checks caught extra scaffold newlines; normalize endings and check new files after staging.

## Decisions not made by kickoff

1. A development-only local venv plus read-only reuse of the existing GPU interpreter avoids reinstalling
   the large stack or disturbing the previous project. The remote setup remains self-contained.
2. Freeze Level-1 holdout by a named seed and SHA-256 ordering; reserve all Level 2. The manifest and
   exact upstream task hashes are committed before sampling. This is a task split, not a selected pilot curriculum.
3. Full C0 requires the target GPU capacity and loaded weights; local preflight has a separate record type.
4. Smoke uses official matrix-multiplication references from the training partition at unmodified input sizes.
   Finite outputs prove execution only, not candidate correctness, timing quality, or C1 isolation.
5. Keep later modules explicitly disabled until the preceding gates pass. No speculative reward/training adapter.
6. Preserve all raw runs outside Git, with passing gate evidence copied to tracked `evidence/gates/`.

## Historical C0 preparation blockers

1. Operator supplied the AutoDL endpoint and explicitly authorized an unlimited AutoDL budget on 2026-10-04.
   The node now has the planned GPU and expanded disk, and C0 passed; do not ask again
   for the already granted budget authorization. C4 go and gate/run requirements remain.
2. C1 isolation and hostile probes are the next unimplemented prerequisite. C0 verifies
   the stack/model/reference path only; it does not authorize unsandboxed candidate execution.
3. Before reward design, complete the kickoff's KernelBench/Kevin/CUDA-L1/Sakana/METR reading; then implement
   C1 on the isolated target environment. Do not run hostile GPU tests on this shared desktop.
4. Before submission, independently verify paper-track word limit, criteria and model eligibility. The official
   prediction-track overview confirms the paper deadline, but the paper-track page body was not readable here.

## Exact resume commands

Reproduce local preparation (creates a fresh immutable record; optional GPU smoke):

```bash
cd /mnt/fastdata/cuda-gemma4-llm
.venv/bin/python -m tools.preflight --gpu-python /mnt/fastdata/glomergence/.venv/bin/python
```

After an approved node is available and the repository is copied there:

```bash
cd /root/autodl-tmp/cuda-gemma4-llm
bash scripts/run_autodl.sh c0
```

## Roadmap

C0–C6 and R0–R3 have completed. The active follow-up is the retrospective
reliability audit, measurement repair/replay, validated custom-kernel contract,
and prospective new-task/API-scaffold comparison described at the top of this
file and in `experiments/reliability-v2-draft.yaml`. The old study is immutable;
the new task manifest and evaluator must be validated and sealed before runs.

## AutoDL image inspection (2026-10-04; read-only, no-GPU mode)

Operator authorized the AutoDL budget without a spending cap and supplied SSH access.
Connected successfully; no password is recorded in project files. Raw evidence:
`runs/inventory/autodl-no-gpu-20261004T200536Z.json`. Reproducible inspection code:
`tools/inspect_image.py` (stdlib only; use the image's Python and CUDA PATH).

| Item | Observed |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 5.15.0-78-generic |
| Image Python | 3.12.3, `/root/miniconda3/bin/python` |
| PyTorch | 2.12.1+cu130; CUDA build 13.0; Triton 3.7.1 |
| CUDA toolkit | `/usr/local/cuda-13.0`, nvcc 13.0.88; CUDA headers present |
| Host compiler | g++ 11.4.0 |
| Host NVIDIA kernel module | 595.58.03 from `/proc/driver/nvidia/version` |
| GPU access | No `/dev/nvidia*`; torch sees zero devices; `nvidia-smi` returns permission denied |
| Actual current CPU/RAM quota | `cpu.max=50000 100000` (0.5 CPU); `memory.max=2147483648` (2 GiB) |
| Filesystems | System overlay 30G; data disk `/root/autodl-tmp` 50G, initially empty |
| Missing tools/packages | uv, ninja, vLLM, transformers, TRL, PEFT, safetensors, ModelScope |

The visible Xeon Platinum 8470Q host has much greater CPU/RAM resources, but those
are **not this container's allocation**. Inventory now records cgroup limits too.
The default noninteractive SSH PATH omits Python and nvcc; explicit paths work.

Assessment: keep this image. Its actual CUDA 13 development toolkit is suitable
as the starting point. Install the project's pinned torch 2.13.0 / vLLM 0.30.0 /
TRL 1.14.0 stack in a separate uv Python environment, preserving the image's Conda
environment. NVIDIA documents CUDA 13.x's minimum driver as >=580; the observed
host module meets that minimum, but GPU attachment, runtime/JIT compatibility and
actual model serving still require C0 on the GPU boot.

Recommendation for the next GPU boot: expand the data disk to about 200 GB for both
models, package/build caches and checkpoints; recheck assigned CPU/RAM and GPU VRAM.
The current 2 GiB no-GPU allocation is inadequate for model loading/full validation.
No packages were installed or image settings changed during this inspection.

Source: [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Formal C0–C6 statuses remain **not run**.

## AutoDL deployment preparation (2026-10-04)

Operator confirmed 200 GB data-disk expansion on the next boot and requested code
preparation. Added `scripts/run_autodl.sh prepare|status|c0`, data-disk cache layout,
image-Python uv bootstrap, Python-header validation, ninja in the GPU dependency
set, pre-download resource checks, and an offline transfer bundle generator.

The CPU stage installs only development dependencies and runs preflight. The GPU
stage checks readiness, installs the pinned stack, downloads/validates both models,
runs C0 once, and stops. Infrastructure readiness is recorded separately from gates.
Package artifacts have SHA-256 manifests and include the pinned upstream checkout.
Deployed to `/root/autodl-tmp/cuda-gemma4-llm` on branch `autodl` using the source
bundle and upstream archive. All transfer hashes verified before extraction.

Remote `prepare` completed successfully with **33 tests passed**. uv 0.11.6 and
managed CPython 3.12.13 are stored under the project's data-disk `.cache/`; the
image Conda environment remains unchanged. The development venv contains only
the CPU dependencies. KernelBench revision and split validation passed offline.

Evidence on the node:

- Successful setup: `runs/setup/prepare-20261004T201721-1488.log`.
- CPU preflight: `runs/preflight/20261004T201734.732216Z/preflight.json` and `tests.xml`.
- Deliberate no-GPU guard check: `runs/preflight/node-readiness-20261004T201810.163890Z.json`.
- First setup failure (archive ownership, fixed): `runs/setup/prepare-20261004T201610-1409.log`.

These records are backed up locally under `runs/autodl-deploy/20261004/runs/`.
The guard exited before large installs, model downloads or formal gate attempts
because the node has no GPU, only 2 GiB RAM and its disk is not expanded yet.
ModelScope was reachable; Hugging Face was unreachable from this node. C0 uses
ModelScope downloads and explicit local snapshot directories.

Formal C0–C6 remain **not run**. No performance/correctness research measurements
were made in this deployment session. After GPU attachment and disk expansion:

```bash
cd /root/autodl-tmp/cuda-gemma4-llm
bash scripts/run_autodl.sh c0
```

This command checks readiness again, installs the pinned GPU dependencies,
downloads and validates both Gemma snapshots, runs C0, and stops before C1.

## Expanded-disk prefetch (2026-10-04; complete)

Operator rebooted without GPU, expanded storage and requested downloads before
attaching the GPU. SSH inspection now reports a **250G** data disk (50G base plus
the expansion), while the CPU/RAM cgroup limits remain 0.5 CPU / 2 GiB. The Git
checkout, managed Python and development venv persisted across reboot. After all
downloads and installs, the data disk has **183G available** (`df -h`; exact free
space at the final readiness check: 196,355,919,872 bytes).

The `prefetch` stage completed. GPU wheels are installed in the data-disk venv:
CPython 3.12.13, torch 2.13.0, vLLM 0.30.0, Transformers 5.17.0, TRL 1.14.0,
PEFT 0.21.0, ModelScope 1.40.1 / modelscope-hub 0.4.5. The image Conda environment
was preserved. CUDA toolkit 13.0.88 and g++ 11.4 remain available for the GPU boot.
`uv pip check` passed for **214 packages**; **35 tests passed** after the final install.
The pinned KernelBench revision and committed split were verified again.

Both snapshots are complete under `/root/autodl-tmp/cuda-gemma4-llm/models/`:

| Snapshot | Weight bytes | Tensors | Validation |
|---|---:|---:|---|
| `gemma-4-12B-it` | 23,919,549,408 | 677 | Source SHA-256 and safetensors structure passed |
| `gemma-4-E4B-it` | 15,992,595,884 | 2,130 | Source SHA-256 and safetensors structure passed |

An independent streaming SHA-256 check compared **all 20 snapshot files**, including
tokenizers/configs/templates, with the captured ModelScope source manifest. Every
file matched, and no unfinished range or merge files remained. Both model families
match C0's expected `gemma4_unified` / `gemma4` configurations.

The matching FlashInfer Python/cubin pair, 0.6.18.post1, is installed. The official
1,565,107,285-byte cubin wheel was downloaded on the workstation and transferred
after AutoDL's GitHub request timed out. Both copies matched release SHA-256
`bbacb5b8bbf429e43bf2740bb45551d981f41d0462842e36ee6fb3e3452763ab`.
Setup now checks and reuses `.cache/wheels/`; vLLM and FlashInfer runtime caches
also live on the data disk. GPU-specific JIT compilation remains for the GPU boot.

Download recovery and limits:

- ModelScope range requests retried transient timeouts. After one SSH interruption,
  existing ranges were resumed with hub-only imports and detached processes; logs
  now survive a closed SSH output pipe. A lightweight helper downloaded E4B alongside
  the main job. Both jobs and the final verifier have exited.
- An optional Transformers-only metadata probe was killed with exit 137 in the
  2 GiB container, without producing a result; the precise SIGKILL source is unknown.
  It is recorded in `LEDGER.md`. Full config/tokenizer/tensor loading is deferred to
  the GPU boot. Hash and structural checks do not substitute for model loading.

Evidence on the node (backed up locally under `runs/autodl-prefetch/20261004/runs/`):

- Completed prefetch: `runs/setup/prefetch-20261004T220208-1924.log`.
- Source manifest: `runs/downloads/source-manifest-20261004.json`.
- All-file hashes: `runs/downloads/verified-source-hashes-20261004T223218.746630Z.json`.
- Weight structure: `runs/downloads/gemma-4-12B-it-20261004T222955.238979Z.json`
  and `runs/downloads/gemma-4-E4B-it-20261004T223151.346698Z.json`.
- Cubin installation: `runs/downloads/cubin-20261004T222927.446443Z.json`
  and `runs/setup/offline-cubin-setup-20261004.log`.
- Package freeze: `runs/setup/prefetch-packages-final-20261004.txt`.
- Final download readiness: `runs/preflight/node-readiness-20261004T223222.136102Z.json`.
- Resume / probe record: `runs/setup/download-resume-20261004T220208.133089Z.json`.
- Reproducible hash verifier: `runs/setup/verify_snapshot_hashes.py`.

The node is ready to shut down and reboot with the planned GPU. **C0–C6 remain
not run** (zero formal gate records); no model inference or training was performed.
Research timing, detector coverage, pass rates, P_signal and rollouts/hour remain
unmeasured. No further operator decision is needed for downloads. Next: attach the
planned GPU, recheck GPU/CPU/RAM allocation, and run:

```bash
cd /root/autodl-tmp/cuda-gemma4-llm
bash scripts/run_autodl.sh c0
```

This performs full loading and actual GPU probes, then stops before C1.

<!-- BEGIN GENERATED PREFLIGHT -->
## Local preparation evidence

Source: `runs/preflight/20261004T195308.561523Z/preflight.json`.

Preparation checks: **PASS**, not a C0 verdict.
Pytest: 28 tests, 0 failures, 0 errors.
Locked tasks: 200; Level-1 train: 80; Level-1 held-out: 20; Level-2 held-out: 100.

GPU inventory: `NVIDIA GeForce RTX 5070 Ti, GPU-ff7b1db6-0062-4462-6dad-1133127093ef, 16303, 561, 595.71.05, 12.0`.
Logical CPUs: 16; free disk: 604.2 GiB.

| Gate | Status | JSON record |
|---|---|---|
| C0 | PASS | `runs/gates/c0-20261004T224213.417987Z-7b8a8dac.json` |
| C1 | not run | — |
| C2 | not run | — |
| C3 | not run | — |
| C4 | not run | — |
| C5 | not run | — |
| C6 | not run | — |

### Numbers and limits

Trusted CUDA reference smoke: 2 modules executed with finite outputs; torch 2.13.0+cu130 / CUDA 13.0.
These executions validate reference plumbing only; no candidate correctness or speedup was measured.
Timing noise floor: not measured. Detector coverage: not measured (no cheats run).
Base-model pass@1, pass@8, P_signal and rollouts/hour: not measured.

### Reachability

- https://huggingface.co: reachable=True, HTTP=200, error=none (does not prove weight-download access).
- https://modelscope.cn: reachable=True, HTTP=200, error=none (does not prove weight-download access).

<!-- END GENERATED PREFLIGHT -->

<!-- BEGIN GENERATED GATES -->
| Gate | Status | Record |
|---|---|---|
| C0 | PASS | `runs/gates/c0-20261006T073539.012183Z-ab7436dc.json` |
| C1 | PASS | `runs/gates/c1-20261006T073631.139042Z-97f9e441.json` |
| C2 | PASS | `runs/gates/c2-20261006T073640.185367Z-834232cf.json` |
| C3 | PASS | `runs/gates/c3-20261006T073735.725497Z-bc623fb4.json` |
| C4 | PASS | `runs/gates/c4-20261005T032853.993235Z-2fac6866.json` |
| C5 | PASS | `runs/gates/c5-20261005T035544.370523Z-bb47b38c.json` |
| C6 | PASS | `runs/gates/c6-20261005T165633.936013Z-88fa5e5d.json` |
| R0 | PASS | `runs/gates/r0-20261006T073423.727861Z-72e40621.json` |
| R1 | FAIL | `runs/gates/r1-20261006T071355.316851Z-b348729c.json` |
| R2 | not run | — |
| R3 | not run | — |

C2: CV across 30 median estimates = 0.0196%; maximum within-pair raw CV = 1.12%. Raw samples were retained, including outliers.
C3: 4/4 deliberate cheats pass naive verification and are rejected by the hardened evaluator.
C4 gemma-4-12B-it: mean P_signal = 0.9880; tasks with nonzero pass@8 = 1.
C4 gemma-4-E4B-it: mean P_signal = 0.8515; tasks with nonzero pass@8 = 1.
<!-- END GENERATED GATES -->
