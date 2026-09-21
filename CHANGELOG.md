# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0](https://github.com/SongshGeoLab/yr-water-quota/compare/v0.2.0...v0.3.0) (2026-09-21)


### ⚠ BREAKING CHANGES

* **model:** 社会项换回乘性式(3)，参数正名 grid/group
* **config:** :boom: AppEff 50 -> 90，canonical 换成 multirun/appeff_090（#62、#95、#64）
* **config:** :boom: 情景组单源化，仓库内那份副本删除（#79）
* **cwatqim:** `s_enforcement_cost` 0.74 -> 0.56、`s_reputation` 0.55 -> 0.53， `WVS_GRID_IS_INVERTED` 语义反转，`load_countries` 改为单一来源（去掉 `results` 形参 与 `grid_alt`/`group_alt` 两列——它们从未被消费）。模型输出会变，需全量重跑。
* **cwatqim:** `City.step` 的社会学习指标由 `payoff` 改为 `unit_payoff`。
* **cwatqim:** `City.hate_a_behave(behave, draw)` 换成 `City.will_report(behave)`； `City.draw_judgements` 删除；`CWatQIModel.step` 新增一次 `snapshot_decision` 广播。
* **cwatqim:** `City.water_withdraw` 去掉 `ga_kwargs`、新增 `kink`； `optimize_surface_share` 更名为 `solve_surface_share` 且不再接受 rng。
* **cwatqim:** `City.agg_payoff` 新增必填参数 `revenue`；效用的函数形式改变， 所有含社会项的结果都会移动。
* **cwatqim:** :boom: β₁ 按 #101 翻转，s_enforcement_cost 0.26 -> 0.74

### Features

* **analysis:** :sparkles: 把地级市集体主义指数对齐到 59 个城市主体 ([e80aeac](https://github.com/SongshGeoLab/yr-water-quota/commit/e80aeac463976a7d95241b562a2c203d6f0c7bf2))
* **config:** :boom: AppEff 50 -&gt; 90，canonical 换成 multirun/appeff_090（[#62](https://github.com/SongshGeoLab/yr-water-quota/issues/62)、[#95](https://github.com/SongshGeoLab/yr-water-quota/issues/95)、[#64](https://github.com/SongshGeoLab/yr-water-quota/issues/64)） ([45e35b8](https://github.com/SongshGeoLab/yr-water-quota/commit/45e35b8baea90d7225bbd1e9a4188df5fb907684))
* **config:** :chart_with_upwards_trend: λ 由扫描定为 0.5，并记下"社会项现在推得动模型" ([f7fa6cf](https://github.com/SongshGeoLab/yr-water-quota/commit/f7fa6cfff2da489e8c1550523b2d29a205aca589))
* **cwatqim:** :boom: β₁ 按 [#101](https://github.com/SongshGeoLab/yr-water-quota/issues/101) 翻转，s_enforcement_cost 0.26 -&gt; 0.74 ([042cab4](https://github.com/SongshGeoLab/yr-water-quota/commit/042cab4ca29a1767a36fa3cf988bf0b6648464cb))
* **cwatqim:** :boom: β₁ 换源并撤销 [#101](https://github.com/SongshGeoLab/yr-water-quota/issues/101) 的翻转，0.74 -&gt; 0.56（[#102](https://github.com/SongshGeoLab/yr-water-quota/issues/102)、[#126](https://github.com/SongshGeoLab/yr-water-quota/issues/126)） ([beaefaa](https://github.com/SongshGeoLab/yr-water-quota/commit/beaefaa09886dafe6d22dce3efa637294f749802))
* **cwatqim:** :boom: 批评改成决策，β₁ 接到执法的外延边际上（[#110](https://github.com/SongshGeoLab/yr-water-quota/issues/110)、[#72](https://github.com/SongshGeoLab/yr-water-quota/issues/72)） ([592e3db](https://github.com/SongshGeoLab/yr-water-quota/commit/592e3db5acf0d46d3c7cab99b47c29dc6bee5473))
* **cwatqim:** :boom: 效用改成加性 U = e − λ·R·(1−s)，社会项的幅度进决策 ([fa37e68](https://github.com/SongshGeoLab/yr-water-quota/commit/fa37e680083049876309df10bf2e77a0be133bc1))
* **cwatqim:** :boom: 社会学习改比单位毛收入的效用，性状不再按城市规模选择 ([b959666](https://github.com/SongshGeoLab/yr-water-quota/commit/b9596660988680fdac3b10ba952caf2e6499966f)), closes [#110](https://github.com/SongshGeoLab/yr-water-quota/issues/110)
* **cwatqim:** :sparkles: Group 参数逐城取值，κ=0 时逐位退回全国标量 ([dfc8192](https://github.com/SongshGeoLab/yr-water-quota/commit/dfc8192308d46fba7897842c49ac8635ccf82d2f))
* **cwatqim:** :sparkles: 给已删除的公开 API 补垫片与去向说明（[#130](https://github.com/SongshGeoLab/yr-water-quota/issues/130)） ([3c54ce9](https://github.com/SongshGeoLab/yr-water-quota/commit/3c54ce901257664a50550e8ed90ff4907b40ea02))
* 地级市集体主义指数——数据管道、逐城 β₂、描述性对比与外部检验 ([61bb53a](https://github.com/SongshGeoLab/yr-water-quota/commit/61bb53a92ff05bb3b72313f903485eceb618fee4))
* 文化参数下沉到省级、协作网络可接入，并换 canonical（[#220](https://github.com/SongshGeoLab/yr-water-quota/issues/220)、[#139](https://github.com/SongshGeoLab/yr-water-quota/issues/139)） ([70bfb13](https://github.com/SongshGeoLab/yr-water-quota/commit/70bfb13fa8ba7e7964f5e22105011863771bc25e))


### Bug Fixes

* **analysis:** :bug: 两轴代码审查的修复：崩溃的 CLI、缺失的行数守卫、文档欠账 ([de3c03e](https://github.com/SongshGeoLab/yr-water-quota/commit/de3c03e3d08a7a7f91f29765450808a6fdc9cfc1))
* **analysis:** :bug: 意图列配对前强制对齐，对不齐就报错（[#120](https://github.com/SongshGeoLab/yr-water-quota/issues/120)） ([b40e706](https://github.com/SongshGeoLab/yr-water-quota/commit/b40e706e6e2907f9534d90137c66cc569f07dc41))
* **cwatqim:** :bug: 三处守卫拦不住 NaN，会让主体静默地再也学不到东西 ([e5ab7d3](https://github.com/SongshGeoLab/yr-water-quota/commit/e5ab7d3f206778a00748c531cf9541bf7f92f317))
* **cwatqim:** :bug: 手滑写 `-m` 时当场说人话，而不是崩在 aquacrop 的 ImportError（[#103](https://github.com/SongshGeoLab/yr-water-quota/issues/103)） ([719738d](https://github.com/SongshGeoLab/yr-water-quota/commit/719738d1af18d7f275fab230b44c9e20337ad996))
* **model:** :bug: 首年当 spin-up 跑掉，分析侧统一丢弃（[#205](https://github.com/SongshGeoLab/yr-water-quota/issues/205)） ([75a9c37](https://github.com/SongshGeoLab/yr-water-quota/commit/75a9c378b47402c1b4f14f31c3141d9db13359b6))
* P0 收尾——正文真源归位、意图列对齐、图的产出口（[#118](https://github.com/SongshGeoLab/yr-water-quota/issues/118) [#119](https://github.com/SongshGeoLab/yr-water-quota/issues/119) [#120](https://github.com/SongshGeoLab/yr-water-quota/issues/120) [#124](https://github.com/SongshGeoLab/yr-water-quota/issues/124) [#137](https://github.com/SongshGeoLab/yr-water-quota/issues/137) [#146](https://github.com/SongshGeoLab/yr-water-quota/issues/146) [#152](https://github.com/SongshGeoLab/yr-water-quota/issues/152) [#86](https://github.com/SongshGeoLab/yr-water-quota/issues/86) [#98](https://github.com/SongshGeoLab/yr-water-quota/issues/98)） ([6b90305](https://github.com/SongshGeoLab/yr-water-quota/commit/6b90305fdf26ef5c01c271b7128cfa990e5fb165))
* 公开 API 垫片、死代码收尾，以及一个测试全绿却毁掉产物的回归（[#130](https://github.com/SongshGeoLab/yr-water-quota/issues/130) [#26](https://github.com/SongshGeoLab/yr-water-quota/issues/26)） ([a3b30dc](https://github.com/SongshGeoLab/yr-water-quota/commit/a3b30dc3a2d17920b252b2c082ee01aa1f85891c))
* 配置单源化与工具链收尾（[#79](https://github.com/SongshGeoLab/yr-water-quota/issues/79) [#140](https://github.com/SongshGeoLab/yr-water-quota/issues/140) [#142](https://github.com/SongshGeoLab/yr-water-quota/issues/142) [#90](https://github.com/SongshGeoLab/yr-water-quota/issues/90)） ([4465c96](https://github.com/SongshGeoLab/yr-water-quota/commit/4465c961f9f6391795f0631d27c89b3e30ca692e))


### Code Refactoring

* **config:** :boom: 情景组单源化，仓库内那份副本删除（[#79](https://github.com/SongshGeoLab/yr-water-quota/issues/79)） ([0f60824](https://github.com/SongshGeoLab/yr-water-quota/commit/0f608241db25a5efa86eb4def0cfd13d040fb09e))
* **cwatqim, analysis:** :recycle: 举报规则收成唯一定义，删掉分析层的第二份实现（[#129](https://github.com/SongshGeoLab/yr-water-quota/issues/129)） ([c619772](https://github.com/SongshGeoLab/yr-water-quota/commit/c619772d023f71ecace83f2a8b44c7f78ef4864f))
* **cwatqim, analysis:** :recycle: 收掉 [#136](https://github.com/SongshGeoLab/yr-water-quota/issues/136) 剩下的四条重复与误导命名 ([e96df83](https://github.com/SongshGeoLab/yr-water-quota/commit/e96df83f2f90d61550d92beb91eefd98b3b398fd))
* **cwatqim:** :recycle: 配水改成两角点闭式求解，删掉差分进化（[#94](https://github.com/SongshGeoLab/yr-water-quota/issues/94)） ([12eb2d9](https://github.com/SongshGeoLab/yr-water-quota/commit/12eb2d9fbb1bf181d55b93a15fef54c2ee00ec69))
* **model:** 社会项换回乘性式(3)，参数正名 grid/group ([eb42bda](https://github.com/SongshGeoLab/yr-water-quota/commit/eb42bdaf1064c9efbf5c85251130889abf8a900c))


### Documentation

* **cwatqim:** :memo: `unit_payoff` 的 docstring 改成如实描述（[#133](https://github.com/SongshGeoLab/yr-water-quota/issues/133) 判定保留现状） ([025199f](https://github.com/SongshGeoLab/yr-water-quota/commit/025199fa96f940ec5595db9a8d28ec85e5251ccc))
* **cwatqim:** :memo: 修掉 /code-review 抓出的四条：矛盾的注释、过期的 docstring、错的性质、瞎的指纹 ([3001bb2](https://github.com/SongshGeoLab/yr-water-quota/commit/3001bb22a413357f2ea8809cec1e370edf948ec7))
* **odd:** :memo: ODD+D 按新的社会机制回写（[#94](https://github.com/SongshGeoLab/yr-water-quota/issues/94)、[#110](https://github.com/SongshGeoLab/yr-water-quota/issues/110)、[#72](https://github.com/SongshGeoLab/yr-water-quota/issues/72)、[#109](https://github.com/SongshGeoLab/yr-water-quota/issues/109)） ([428291a](https://github.com/SongshGeoLab/yr-water-quota/commit/428291a77925d1197a300f0fdd954b5264242aae))
* 把 grid/group 的语义与新标定同步进面向读者的文档 ([7036907](https://github.com/SongshGeoLab/yr-water-quota/commit/70369070c1dcb3023af4be12a500920eb0084dd3))

## [0.2.0](https://github.com/SongshGeoLab/yr-water-quota/compare/v0.1.6...v0.2.0) (2026-08-18)


### ⚠ BREAKING CHANGES

* **cwatqim:** `cwatqim` 移除了四个公开名字：`cwatqim.core.payoff.check_boundary`、 `cwatqim.core.data_loaders.load_quotas`、`City.get_cells`、`City.climate_datapath`。 四者在本仓零调用方；`City.climate_datapath` 的替代是 `City.climate_data`。
* **cwatqim:** `ureg` 不再从 `cwatqim.core` 导出，`from cwatqim.core import ureg` 会失败。它是一个全仓从未使用的 pint UnitRegistry。ddd2286 用了 `refactor:` 而没加 `!`，release-please 因此不会把这次移除记进 CHANGELOG； 本条补上标记。

### Features

* **cwatqim:** :sparkles: 补齐效用与决策的采集口径（[#70](https://github.com/SongshGeoLab/yr-water-quota/issues/70)） ([a54ff57](https://github.com/SongshGeoLab/yr-water-quota/commit/a54ff5790cdd0bf3f23ec593fbffd31b0e578486))


### Bug Fixes

* **analysis:** :bug: 修掉恒假分支、stdout 劫持与 __all__ 未导入名（[#22](https://github.com/SongshGeoLab/yr-water-quota/issues/22)） ([bf56967](https://github.com/SongshGeoLab/yr-water-quota/commit/bf56967cf89def608338de537adfd3f7f7c6c629))
* **analysis:** :wrench: 顶层配置键 reports 迁移到 tracker（[#35](https://github.com/SongshGeoLab/yr-water-quota/issues/35)） ([941bbe3](https://github.com/SongshGeoLab/yr-water-quota/commit/941bbe3797a3e6c9afaa0bde6d014c428bd4bd06))
* **config:** :wrench: multirun 目录按情景命名，并纠正被我说大的口径（[#71](https://github.com/SongshGeoLab/yr-water-quota/issues/71)） ([43b4a42](https://github.com/SongshGeoLab/yr-water-quota/commit/43b4a42f7a9d53e0d1a7a2de836849c67068073f))
* **cwatqim:** :bug: include_s 恢复年份门槛，社会项不再全程参与 payoff（[#42](https://github.com/SongshGeoLab/yr-water-quota/issues/42)） ([a6b7968](https://github.com/SongshGeoLab/yr-water-quota/commit/a6b7968cec18b267d01bdfcf291383b838f30528))
* **cwatqim:** :bug: multirun + 固定种子不再崩（[#82](https://github.com/SongshGeoLab/yr-water-quota/issues/82)） ([3a41541](https://github.com/SongshGeoLab/yr-water-quota/commit/3a415414c53cbeff4f1d401e52345a251b98143d))
* **cwatqim:** :bug: 决策倾向每年刷新，政策年份缺失时报错 ([914f95f](https://github.com/SongshGeoLab/yr-water-quota/commit/914f95ff5539be520175aee43fba8750afa1a545)), closes [#16](https://github.com/SongshGeoLab/yr-water-quota/issues/16)
* **cwatqim:** :bug: 同侪判定的随机数每年抽一次，不在目标函数里抽（[#61](https://github.com/SongshGeoLab/yr-water-quota/issues/61)） ([ab47d6c](https://github.com/SongshGeoLab/yr-water-quota/commit/ab47d6cd01445fc87d29e2a3ed48963adce835cf))
* **cwatqim:** :bug: 惰性导出，修好 python -m cwatqim（[#48](https://github.com/SongshGeoLab/yr-water-quota/issues/48)） ([e2d29dc](https://github.com/SongshGeoLab/yr-water-quota/commit/e2d29dcc9e4921d3ce172e2c4148f442b14bddb2))
* **cwatqim:** :bug: 拒绝 forced_since 早于 include_s_since 的情景（[#59](https://github.com/SongshGeoLab/yr-water-quota/issues/59)） ([4dd3f1a](https://github.com/SongshGeoLab/yr-water-quota/commit/4dd3f1a99502942aa901260ffb86dfc012435b2d))
* **cwatqim:** :bug: 缓存键补上 city_id，并纠正上游病因的说法（[#68](https://github.com/SongshGeoLab/yr-water-quota/issues/68)） ([e5c4990](https://github.com/SongshGeoLab/yr-water-quota/commit/e5c49900d07d0ecbfa719bc51de4029d665b8d54))
* **cwatqim:** :bug: 补上 CWatQIModel.end() 的落盘实现（[#17](https://github.com/SongshGeoLab/yr-water-quota/issues/17)） ([cc8dc76](https://github.com/SongshGeoLab/yr-water-quota/commit/cc8dc7668dce3a6052b7012be0be6881bafd66c6))
* **cwatqim:** :bug: 配水优化补传 crop_prices，目标函数恢复作物收益（[#15](https://github.com/SongshGeoLab/yr-water-quota/issues/15)） ([1f009ae](https://github.com/SongshGeoLab/yr-water-quota/commit/1f009ae25f05be339b6e408f0f2d27db131e639b))
* **cwatqim:** :bug: 零取水时灌溉上限返回 0 而不是除零（[#66](https://github.com/SongshGeoLab/yr-water-quota/issues/66)） ([d7fb26a](https://github.com/SongshGeoLab/yr-water-quota/commit/d7fb26acab72f528572da862d3e60018f5a32c08))
* **cwatqim:** :memo: 社会项改名为 social_standing，纠正反号的文档（[#60](https://github.com/SongshGeoLab/yr-water-quota/issues/60)） ([96c05d1](https://github.com/SongshGeoLab/yr-water-quota/commit/96c05d1b577a1218cae95e88bb5e43a8c71f748c))
* **cwatqim:** :sparkles: 采集 e / s / payoff 三个收益变量（[#63](https://github.com/SongshGeoLab/yr-water-quota/issues/63)） ([34ee86c](https://github.com/SongshGeoLab/yr-water-quota/commit/34ee86c7250800499909bfa8fd8c4a17bf3455fb))
* **cwatqim:** 决策倾向每年刷新，政策年份缺失时报错 ([#16](https://github.com/SongshGeoLab/yr-water-quota/issues/16)) ([4e49cda](https://github.com/SongshGeoLab/yr-water-quota/commit/4e49cdac132d23b7a01acc18fd1ad74b5c9b0310))
* multirun + 固定种子直接崩（[#82](https://github.com/SongshGeoLab/yr-water-quota/issues/82)） ([a9f114f](https://github.com/SongshGeoLab/yr-water-quota/commit/a9f114ff99f05a48ffcae0cbaf9cf35f6cc66c39))
* P1 批次 —— include_s 年份门槛、口径澄清、重复实现与分层（[#42](https://github.com/SongshGeoLab/yr-water-quota/issues/42) [#38](https://github.com/SongshGeoLab/yr-water-quota/issues/38) [#39](https://github.com/SongshGeoLab/yr-water-quota/issues/39) [#24](https://github.com/SongshGeoLab/yr-water-quota/issues/24) [#23](https://github.com/SongshGeoLab/yr-water-quota/issues/23) [#43](https://github.com/SongshGeoLab/yr-water-quota/issues/43)） ([f155ccf](https://github.com/SongshGeoLab/yr-water-quota/commit/f155ccf78d19c1107f3d3f53adbd40ad052f5fd1))
* 修完剩余四个 P0（[#17](https://github.com/SongshGeoLab/yr-water-quota/issues/17) [#19](https://github.com/SongshGeoLab/yr-water-quota/issues/19) [#20](https://github.com/SongshGeoLab/yr-water-quota/issues/20) [#22](https://github.com/SongshGeoLab/yr-water-quota/issues/22)） ([6720fc0](https://github.com/SongshGeoLab/yr-water-quota/commit/6720fc0cf007abca20950e49ffbc9423f584276b))
* 对照审稿意见复核代码，修 7 处缺陷（[#58](https://github.com/SongshGeoLab/yr-water-quota/issues/58) [#59](https://github.com/SongshGeoLab/yr-water-quota/issues/59) [#60](https://github.com/SongshGeoLab/yr-water-quota/issues/60) [#61](https://github.com/SongshGeoLab/yr-water-quota/issues/61) [#63](https://github.com/SongshGeoLab/yr-water-quota/issues/63) [#66](https://github.com/SongshGeoLab/yr-water-quota/issues/66) [#73](https://github.com/SongshGeoLab/yr-water-quota/issues/73)） ([b2dc727](https://github.com/SongshGeoLab/yr-water-quota/commit/b2dc7271a52fc8a52a0de3a5ec3dc7b1b3566569))
* 水价文件对不上（[#49](https://github.com/SongshGeoLab/yr-water-quota/issues/49)）、地下水趋势静默降级（[#25](https://github.com/SongshGeoLab/yr-water-quota/issues/25)），以及一批死代码与文档幽灵（[#26](https://github.com/SongshGeoLab/yr-water-quota/issues/26)） ([08e6539](https://github.com/SongshGeoLab/yr-water-quota/commit/08e653946bd8fd09d59a1828424ba72c1e52b8cf))
* 清理全部六个 P2（[#28](https://github.com/SongshGeoLab/yr-water-quota/issues/28) [#29](https://github.com/SongshGeoLab/yr-water-quota/issues/29) [#30](https://github.com/SongshGeoLab/yr-water-quota/issues/30) [#31](https://github.com/SongshGeoLab/yr-water-quota/issues/31) [#32](https://github.com/SongshGeoLab/yr-water-quota/issues/32) [#35](https://github.com/SongshGeoLab/yr-water-quota/issues/35)） ([1007d6a](https://github.com/SongshGeoLab/yr-water-quota/commit/1007d6a8e02a31e59e4d8a6da1a1637fc0b70896))
* 配水优化补传 crop_prices，目标函数恢复作物收益（[#15](https://github.com/SongshGeoLab/yr-water-quota/issues/15)） ([1d9f916](https://github.com/SongshGeoLab/yr-water-quota/commit/1d9f916035dfe19624e67b0f8c5e8d0948b98e13))


### Performance Improvements

* **cwatqim:** :zap: dynamic variable 每年只读一次（[#68](https://github.com/SongshGeoLab/yr-water-quota/issues/68)） ([61d9d5e](https://github.com/SongshGeoLab/yr-water-quota/commit/61d9d5e82a057b6760fdf0ac8abcbac773ffa21b))
* **cwatqim:** :zap: 社会项提出目标函数，一年算两次（[#72](https://github.com/SongshGeoLab/yr-water-quota/issues/72) 的 a+b） ([94374fa](https://github.com/SongshGeoLab/yr-water-quota/commit/94374faf0c97a5c3492dda9daf6b6c96c0860422))


### Code Refactoring

* **config:** :wrench: 情景做成 hydra config group（[#71](https://github.com/SongshGeoLab/yr-water-quota/issues/71)） ([9fe9f44](https://github.com/SongshGeoLab/yr-water-quota/commit/9fe9f44fd1454ff4001c0628118956dae5f562fe))
* **cwatqim:** :fire: 删掉 pint 直接依赖（ureg 已移除） ([cc3a277](https://github.com/SongshGeoLab/yr-water-quota/commit/cc3a2770f54ae8877ccd0c955dcc1aa2e6e6642c))
* **cwatqim:** :fire: 删掉四处零调用方的死代码（[#26](https://github.com/SongshGeoLab/yr-water-quota/issues/26)） ([64ccdbd](https://github.com/SongshGeoLab/yr-water-quota/commit/64ccdbdf6a0e932c67a4d14d7f55a5e8338be050))
* **cwatqim:** :fire: 删掉死代码 ureg 与四份多余的 TypeAlias shim（[#26](https://github.com/SongshGeoLab/yr-water-quota/issues/26)） ([ddd2286](https://github.com/SongshGeoLab/yr-water-quota/commit/ddd228623b92e765c8f1f4cc90dea9617b6a07fb))
* **cwatqim:** :recycle: 把灌溉配比优化从 City 里抽出来（[#27](https://github.com/SongshGeoLab/yr-water-quota/issues/27)） ([3eb7a87](https://github.com/SongshGeoLab/yr-water-quota/commit/3eb7a875d0a668725ac62525a43b0901427e540f))
* **cwatqim:** :recycle: 收拾 [#48](https://github.com/SongshGeoLab/yr-water-quota/issues/48) / [#72](https://github.com/SongshGeoLab/yr-water-quota/issues/72) 的审查意见 ([e32a099](https://github.com/SongshGeoLab/yr-water-quota/commit/e32a099c5196833c792c4baa0964d92d65463f16))
* **cwatqim:** 把灌溉配比优化从 City 里抽出来（[#27](https://github.com/SongshGeoLab/yr-water-quota/issues/27)） ([6bfbc49](https://github.com/SongshGeoLab/yr-water-quota/commit/6bfbc498771de6f747bf91c85dd78ddce3f83fb2))
* 死代码清理、ci/ 更名 tracking/、docs notebook 断裂 import（[#26](https://github.com/SongshGeoLab/yr-water-quota/issues/26)） ([eee6878](https://github.com/SongshGeoLab/yr-water-quota/commit/eee687866d4b9a42266cd28e4e4545c89dd05a1e))


### Documentation

* **cwatqim:** :memo: 写清渠系与田间两层灌溉效率，修正 ODD+D（[#62](https://github.com/SongshGeoLab/yr-water-quota/issues/62)） ([98f2482](https://github.com/SongshGeoLab/yr-water-quota/commit/98f248226b23a96867fd4ad68047a69b71b21a9e))
* **cwatqim:** 写清渠系与田间两层灌溉效率，修正 ODD+D（[#62](https://github.com/SongshGeoLab/yr-water-quota/issues/62)） ([fb54113](https://github.com/SongshGeoLab/yr-water-quota/commit/fb54113c138ba49d7fe6c578bd5e42272fee988d))
* **ODD+D:** :memo: 改正社会网络与优化时序的描述（[#67](https://github.com/SongshGeoLab/yr-water-quota/issues/67)、[#77](https://github.com/SongshGeoLab/yr-water-quota/issues/77)） ([7ae9226](https://github.com/SongshGeoLab/yr-water-quota/commit/7ae92267d2dea1cf6c828d970dd9183b0ecc2338))

## [0.1.6](https://github.com/SongshGeoLab/yr-water-quota/compare/v0.1.5...v0.1.6) (2026-01-15)


### Bug Fixes

* **cwatqim:** :wrench: update demo configuration and documentation for improved usability ([29db1c9](https://github.com/SongshGeoLab/yr-water-quota/commit/29db1c91ec27b54249da57c5c8413527c5f78169))

## [0.1.5](https://github.com/SongshGeoLab/yr-water-quota/compare/v0.1.4...v0.1.5) (2026-01-14)


### Bug Fixes

* **cwatqim:** update creator affiliation and clear related identifiers in .zenodo.json ([d651b02](https://github.com/SongshGeoLab/yr-water-quota/commit/d651b02c907a6ffcc9d9ec1798c4110b97427b70))

## [0.1.4](https://github.com/SongshGeoLab/yr-water-quota/compare/v0.1.3...v0.1.4) (2026-01-14)


### Bug Fixes

* **docs:** :memo: add DOI to CITATION.cff, pyproject.toml, and README.md for improved citation tracking ([797d02f](https://github.com/SongshGeoLab/yr-water-quota/commit/797d02fcaa409237c2a5fadca6b48fb4ea4c41af))

## [0.1.3](https://github.com/SongshGeoLab/yr-water-quota/compare/v0.1.2...v0.1.3) (2026-01-14)


### Bug Fixes

* **citation:** :memo: remove ORCID entry from CITATION.cff file ([2e96c13](https://github.com/SongshGeoLab/yr-water-quota/commit/2e96c1304ce78452a1f65be2cb1d74ac16175c8b))

## [0.1.2](https://github.com/SongshGeoLab/yr-water-quota/compare/v0.1.1...v0.1.2) (2026-01-14)


### Bug Fixes

* **ci:** :bug: update release-please workflow to use environment variables for logging output ([b69ef12](https://github.com/SongshGeoLab/yr-water-quota/commit/b69ef1213ece787863eb83335bfbf26dd3f34759))

## [0.1.1](https://github.com/SongshGeoLab/yr-water-quota/compare/v0.1.0...v0.1.1) (2026-01-14)


### Bug Fixes

* **ci:** :bug: improve sync workflow for public repository; add checks for remote branch existence and enhance error handling during subtree pull and push operations ([f446b43](https://github.com/SongshGeoLab/yr-water-quota/commit/f446b431399b4abaa2485457314c157daefe5dcf))
* **ci:** :bug: update release configuration for cwatqim; add package-specific settings and enhance workflow documentation ([16822f1](https://github.com/SongshGeoLab/yr-water-quota/commit/16822f116b95d01947343438873dc5ef28a51785))
* **citation:** :memo: correct indentation for ORCID entry in CITATION.cff file ([acce24d](https://github.com/SongshGeoLab/yr-water-quota/commit/acce24dd9b6a4dfd7ff490c69fe3c17fbe9c832a))
* **metadata:** :memo: update model description and keywords in CWatQIM files; change "multi-agent simulation" to "ABM simulation" and correct author details in CITATION.cff ([4a1b9ce](https://github.com/SongshGeoLab/yr-water-quota/commit/4a1b9ce17516896b1db0a1df606564ff17c181d0))


### Code Refactoring

* **documentation:** enhance package descriptions and docstrings for clarity; improve examples and usage instructions across modules in the CWatQIM framework ([99f402d](https://github.com/SongshGeoLab/yr-water-quota/commit/99f402dea2bcae4f7058fd64ebc752f261f72f2b))
* **project:** remove Nature and update model structure to use CWatQIModel; add main execution script for batch experiments ([2f81153](https://github.com/SongshGeoLab/yr-water-quota/commit/2f811531f52eef75e43d4413c8812aa176fd9e32))
* **project:** seperate the project into model and analysis two parts. ([0e1b86d](https://github.com/SongshGeoLab/yr-water-quota/commit/0e1b86da9a073b91cd42cbc3bd752099e51f73a5))
* **tests:** update test fixtures for CWatQIModel; enhance documentation and improve model instance creation for clarity and consistency in testing ([60e0342](https://github.com/SongshGeoLab/yr-water-quota/commit/60e0342a2494bc6aac8ef062ccbd33d7ed18d6b7))


### Documentation

* **citation:** :memo: add CITATION.cff file for proper software citation and metadata; include author details, abstract, and keywords for CWatQIM ([31135a8](https://github.com/SongshGeoLab/yr-water-quota/commit/31135a800ba86c9f9992ed1c89f0ca5a2fd66a4f))

## [0.1.0] - 2026-01-14

### Added

- Initial release of CWatQIM (Crop-Water Quota Irrigation Model)
- Province-level water resource management agents
- City-level agricultural irrigation agents
- Water quota allocation mechanisms based on Yellow River "87 Agreement"
- Integration with ABSESpy framework for agent-based modeling
- Support for AquaCrop model integration via aquacrop-abses
- Climate data processing from ERA5 reanalysis
- Groundwater and surface water source switching logic
- Payoff calculation for irrigation decisions
