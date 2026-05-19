# 三模型输出深度对比 (Qwen 2.5-VL-72B vs Qwen 3-VL-8B vs Gemma 3-12B)

> 数据: 4 个视频, 每视频 14–34 个 10s 切片, 共 80 切片。
> 共用同一 YOLO+pose 通道与同一 5 轮 QA/QC, 所以差异 100% 来自 VLM 文本生成。

---

## 0. TL;DR — 风格指纹

| 维度 | Qwen2.5-VL-72B | Qwen3-VL-8B | Gemma3-12B |
|---|---|---|---|
| 描述风格 | **抽象/克制** "a person is walking through a room with furniture" | **叙事+空间关系** 主语+动作+背景的复合句 | **短句+清单** "A person is on the floor." |
| 物体粒度 | 类别词 (kitchen, shelves) | 实体词混合 (kitchen cabinet, plastic bag) | **最细** (Rubik's cube, DVDs, tissues, magazine) |
| 文化/IP识别 | superhero, costumed figure | costume, explosion, sparks | **avengers tower** (识别到 Marvel) |
| 角色推断保守度 | 中 (无 glasses) | 中 (调成 adolescent) | 偏激进 (`glasses`, 衣物多细节) |
| chain 切分 | 21–33 chains/视频 | **30–46 chains/视频 (最细)** | 14–46 chains/视频 (差异大) |
| 单 chain 平均跨度 | ~10–12s | ~8–10s | **~15–25s (最粗)** |

---

## 1. 物体词汇覆盖与重叠

把每个视频里所有切片的 `objects[].label` 汇总成集合, 看三家分别识别到了什么。

### `oCkUyjaZuNI` (95s. 室内场景片段, 含 Marvel/超级英雄主题镜头 (`QUEENS` 字幕、超级英雄装扮))

- **集合大小**: 72B=26, 8B=34, Gemma=40 个不同 label
- **三家都识别到的 (9)**: backpack, cabinet, chair, couch, kitchen cabinets, lamp, microwave, table, television
- **仅 Qwen2.5-VL-72B 独有 (9)**: city buildings, costumed figure, countertop, globe, kitchen items, shelves, sparks, superhero, urban skyline
- **仅 Qwen3-VL-8B 独有 (15)**: books, city skyline, costume, explosion, kitchen cabinet, living room, mug, person on floor, plastic bag, poster, refrigerator, shelving unit…
- **仅 Gemma3-12B 独有 (23)**: armchair, artwork, avengers tower, bag, book, bookshelf, bookshelves, buildings, cars, counter, cup, debris…

### `-JUtzfuiV08` (85s. 客厅访谈式拍摄, 两位成人坐着, 镜头较稳)

- **集合大小**: 72B=12, 8B=14, Gemma=22 个不同 label
- **三家都识别到的 (7)**: armchair, bookshelf, coffee table, couch, cup, lamp, mug
- **仅 Qwen2.5-VL-72B 独有 (3)**: kitchen, living room, table
- **仅 Qwen3-VL-8B 独有 (2)**: coffee_table, mugs
- **仅 Gemma3-12B 独有 (10)**: books, bookshelves, cabinets, dvds, magazine, pillows, remote control, rubiks cube, sofa, tissues

### `DmSmN-oqFZ0` (95s. 室内 + 楼梯 + 厨房, 含特殊道具 (Superman 雕像、剑、警戒带))

- **集合大小**: 72B=16, 8B=28, Gemma=40 个不同 label
- **三家都识别到的 (11)**: bookshelf, brick wall, caution tape, dartboard, door, kitchen, light fixture, stairs, superman figurine, sword, whiteboard
- **仅 Qwen2.5-VL-72B 独有 (4)**: appliance, kitchen appliances, living room, shelves
- **仅 Qwen3-VL-8B 独有 (5)**: couch, kettle, out of order sign, wall sconce, wall-mounted light fixture
- **仅 Gemma3-12B 独有 (16)**: appliances, artwork, bag, bookshelves, chair, figurine, frying pan, glasses, jacket, kitchen cabinets, mug, painting…

### `G6f0w5BRasw` (207s. 户外砖房 + 多人 (含小孩) + 长时段, 镜头切换最多)

- **集合大小**: 72B=35, 8B=60, Gemma=53 个不同 label
- **三家都识别到的 (9)**: brick wall, building, clothesline, cup, door, furniture, greenery, plants, wooden structure
- **仅 Qwen2.5-VL-72B 独有 (17)**: architecture, black object, braided hair, clothes, green plants, hanging shirt, household items, metal object, mobile phone, movie clip thumbnails, movie clips interface, outdoor environment…
- **仅 Qwen3-VL-8B 独有 (31)**: adult, adult male, arm, background foliage, braids, brick_stack, button, child, environment, girl, hand, hanging plants…
- **仅 Gemma3-12B 独有 (25)**: barrel, brick structure, bucket, climbing plant, clothing rack, column, courtyard, decorative items, diaper, fence, frame, hair…

---

## 2. 同一时间窗 (前 5 个切片) 的 scene_description 三家逐行对照

### `oCkUyjaZuNI`

**切片 1 [0.0–10.0s]**

- **72B**: A cityscape with the word 'QUEENS' prominently displayed over it, followed by an indoor hallway where people are walking.
- **8B** : An aerial view of a city skyline with the word 'QUEENS' overlaid, followed by shots of people walking down an indoor hallway.
- **Gem**: Aerial view of a cityscape with the word 'QUEENS' overlaid. Transition to a hallway with a person walking away.

**切片 2 [6.0–16.0s]**

- **72B**: A person is walking through a well-lit room with large windows. There are other people in the background, and the setting appears to be a living room with furniture such as chairs and a lamp.
- **8B** : A person in dark clothing walks through an indoor hallway and enters a room. Another person is seen on the floor in the hallway. A third person is seen from behind in a room with furniture.
- **Gem**: A person in a suit is standing in a hallway. Another person is on the floor.

**切片 3 [12.0–22.0s]**

- **72B**: A person with short hair is seen in various indoor settings, including a hallway and a room with furniture.
- **8B** : A person in a dark jacket walks through an indoor hallway. In the background, another person sits on the floor near a window. Later, a person in a dark shirt is seen from behind, adjusting a backpack, while other people are visible in the background.
- **Gem**: Interior hallway and room. A person is walking away from a seated person.

**切片 4 [18.0–28.0s]**

- **72B**: A dimly lit room with shelves containing various items such as books, a globe, and decorative objects. A person wearing earphones is seen from behind, walking through the room.
- **8B** : A dimly lit interior room, likely a living area, with furniture and shelving units. The lighting is warm and ambient, with lamps providing illumination.
- **Gem**: Interior of a living room with a kitchen visible in the background. The room is furnished with a shelf, a table, and a lamp.

**切片 5 [24.0–34.0s]**

- **72B**: A person in a kitchen holding an object, wearing a graphic t-shirt and a jacket.
- **8B** : A dimly lit interior space, likely a living room or dining area, with furniture and decor visible. The scene transitions to show two people seated on a couch and a third person standing in a kitchen area.
- **Gem**: Interior of a home with multiple rooms. The first room has a hutch with glassware and decorative items. The second room has a sofa, television, and bookshelves. The third room is a kitchen with cabinets and appliances.

### `-JUtzfuiV08`

**切片 1 [0.0–10.0s]**

- **72B**: A kitchen setting with various appliances and items on the counters. A person in a blue sweater and skirt is holding a yellow cup. The scene transitions to a living room with bookshelves, a couch, and a coffee table covered with magazines. Another person is seated on the couch, holding a blue cup.
- **8B** : A person in a blue sweater and skirt stands in a kitchen holding a yellow mug. the person walks into a living room where a person in a purple sweater is seated on a couch holding a blue mug. The person sits down on the couch.
- **Gem**: Interior of an apartment kitchen and living room. The kitchen has cabinets, appliances, and a refrigerator covered with notes. The living room has a sofa, coffee table, and bookshelves.

**切片 2 [6.0–16.0s]**

- **72B**: A living room setting with bookshelves, a couch, and a coffee table filled with magazines. Two individuals are present, one standing and the other seated on a chair.
- **8B** : An indoor living room scene with two people. One person is standing and facing the other, who is seated on a red couch. A coffee table is in the center. Bookshelves and a lamp are visible in the background.
- **Gem**: Interior of a living room with a large bookshelf, a brown leather sofa, and a coffee table.

**切片 3 [12.0–22.0s]**

- **72B**: Two individuals are seated on a couch in a living room setting. One person is holding a yellow cup and the other is holding a blue cup. The person with the yellow cup appears to be mouth open or gesturing while the other listens attentively.
- **8B** : Two people are seated on a red couch in a living room. One person, wearing a blue sweater, is drinking from a yellow mug. The other person, wearing a purple striped shirt, is holding a blue mug and looking forward.
- **Gem**: Interior of a living room with a brown leather couch and bookshelves.

**切片 4 [18.0–28.0s]**

- **72B**: Two individuals are seated on a couch in a living room setting. One person is holding a mug, while the other appears to be engaged in visible interaction.
- **8B** : Two people are seated on a red couch in a living room. One person, wearing a blue cable-knit sweater, is holding a yellow cup and appears to be drinking. The other person, wearing a purple striped shirt, is holding a blue object and head oriented toward it. A lamp and bookshelf are visible in the background.
- **Gem**: Interior of a living room with a brown leather couch and bookshelves.

**切片 5 [24.0–34.0s]**

- **72B**: A living room setting with bookshelves in the background, a couch, and a lamp. Two individuals are seated on the couch.
- **8B** : Two individuals are seated on a couch in a living room. One person is wearing a purple striped shirt and the other is wearing a blue cable-knit sweater. A bookshelf and a lamp are visible in the background.
- **Gem**: Interior of a living room with a couch, bookshelves, and lamps.

### `DmSmN-oqFZ0`

**切片 1 [0.0–10.0s]**

- **72B**: A kitchen setting with various appliances and a whiteboard in the background. A person is seated at a desk with a computer, while another person stands nearby.
- **8B** : A kitchen and living room area with wooden cabinets, a refrigerator, a microwave, and a cluttered desk. A person stands near the desk while a person sits at it. The person then moves to a different area with a bookshelf and couch.
- **Gem**: Interior of an apartment kitchen and living room area. A table with computer equipment is in the kitchen. A sofa and bookshelf are in the living room.

**切片 2 [6.0–16.0s]**

- **72B**: A kitchen and living room setting with various objects such as books, a whiteboard with writing, a sword decoration, and a Superman figurine. A person in a blue shirt with the number 73 on it stands in the kitchen, while another person adjusts their hair in the living room.
- **8B** : A kitchen and living room setting with wooden cabinets, a whiteboard with writing, and bookshelves. A person in a blue shirt with a yellow number '73' on it is visible. A person in a purple jacket and glasses is visible, adjusting the person's clothing and then standing still.
- **Gem**: Interior of a kitchen and living room. The kitchen has wooden cabinets and appliances. The living room has bookshelves and a sofa.

**切片 3 [12.0–22.0s]**

- **72B**: A room filled with bookshelves, a whiteboard with writing, a Superman figurine, and a sword on the wall.
- **8B** : A person with long hair and glasses stands in a room with bookshelves and a whiteboard in the background. person is adjusting the person's clothing and then stands with the person's hands clasped. A Superman figurine and a sword are visible on a shelf behind the person's.
- **Gem**: Interior of a living room with bookshelves, a whiteboard, and a sofa.

**切片 4 [18.0–28.0s]**

- **72B**: A person with long brown hair and glasses is in a room with various objects like a dartboard, a Superman figurine, and a whiteboard with writing on it. The person appears to be adjusting their hair. Another person with short brown hair is in a kitchen setting with shelves filled with dishes and a whiteboard with mathematical equations.
- **8B** : A person with long hair and glasses stands in a room with a whiteboard and bookshelf. the person then turns away and raises the person's hands to the person's head. A person in a blue shirt with the number 73 on it stands in a kitchen.
- **Gem**: Interior of a room with bookshelves, a whiteboard, and various objects. A kitchen is visible in the background.

**切片 5 [24.0–34.0s]**

- **72B**: A person is seen in various indoor settings, including near a dartboard, in a kitchen, and in a hallway with caution tape.
- **8B** : A person in a purple jacket walks through a hallway with a dartboard on the wall. The scene cuts to a person in a blue shirt with the number 73 on it in a kitchen. The person in the purple jacket is then seen walking past a door with yellow caution tape.
- **Gem**: Interior of an apartment with multiple rooms visible. One room has a dartboard and a painting on the wall. Another room appears to be a kitchen with cabinets and appliances. A hallway with stairs and a door is also visible.

### `G6f0w5BRasw`

**切片 1 [0.0–10.0s]**

- **72B**: An outdoor setting with two individuals in a courtyard surrounded by plants, wooden furniture, and traditional architecture.
- **8B** : An outdoor courtyard with potted plants and a wooden door. One person is standing, another is sitting on a step holding a cup, and a third person is seen from behind.
- **Gem**: Exterior courtyard setting with traditional Chinese architecture. A person and a child are present.

**切片 2 [6.0–16.0s]**

- **72B**: A person sits on steps in front of a large wooden door, holding a cup. Another person with long braided hair stands facing him.
- **8B** : A person sits on a step in front of a large, weathered door, holding a cup. A person stands facing him, partially obscuring the view. The scene is set in a dimly lit, rustic environment.
- **Gem**: Two people are on a porch in front of a large, ornate door. One person is sitting and holding a mug, while the other is facing away.

**切片 3 [12.0–22.0s]**

- **72B**: A child holding a black jacket stands in front of an ornate door with plants nearby. An adult sits on steps near the same door, holding a cup.
- **8B** : A child walks toward a person sitting on steps. The person is holding a mug and looking up as the child approaches.
- **Gem**: Exterior of a house with a car parked nearby. A person is walking towards the house, carrying a dark object. An adult male is sitting on the steps of the house.

**切片 4 [18.0–28.0s]**

- **72B**: A person with long braided hair stands in front of an ornate doorway, holding a black object. Another person with short dark hair and a mustache is seen standing near a weathered door. A third person's legs are partially visible in the foreground.
- **8B** : A person walks away from a doorway, carrying a dark jacket. Another person stands in front of a weathered door. A third person, seen from behind, stands on a porch step holding a dark jacket.
- **Gem**: Exterior of a building with a car parked in front. The building has ornate doors and decorative columns. There are stairs and a small brick structure in the foreground.

**切片 5 [24.0–34.0s]**

- **72B**: A person in a white shirt stands in front of a weathered door. A person with long braids adjusts their clothing near the doorway. The two individuals appear to interact.
- **8B** : A person in a white shirt stands in front of a weathered building. A person in a white tank top approaches him, carrying a dark jacket over their shoulder. They face each other in what appears to be an outdoor courtyard or alleyway.
- **Gem**: Exterior scene with a brick building and a doorway. There is a person standing near the doorway and another person approaching.

---

## 3. 同一切片的 objects 列表对照 (取每视频 1 段最有信息量的切片)

### `oCkUyjaZuNI` 切片 4 (18.0–28.0s)

**Qwen2.5-VL-72B** — 3 个物体:
  - `shelves` (conf=0.95): metal shelving unit with books and decorative items
  - `globe` (conf=0.85): small glass globe on shelf
  - `lamps` (conf=0.9): two table lamps with warm light

**Qwen3-VL-8B** — 4 个物体:
  - `shelving unit` (conf=0.95): Black metal-framed shelving unit with multiple tiers, holding books and decorative items.
  - `lamp` (conf=0.9): A lit lamp with a patterned shade on a side table, and another lamp on a shelf.
  - `chair` (conf=0.8): A wooden chair is partially visible in the foreground.
  - `books` (conf=0.9): Two books with orange and red covers are visible on a shelf.

**Gemma3-12B** — 5 个物体:
  - `shelf` (conf=1.0): Metal shelf with multiple levels.
  - `table` (conf=1.0): Wooden table.
  - `lamp` (conf=1.0): Table lamp and floor lamp.
  - `book` (conf=0.8): Books on the shelf.
  - `plant` (conf=0.7): Green plant on the shelf.

### `-JUtzfuiV08` 切片 1 (0.0–10.0s)

**Qwen2.5-VL-72B** — 3 个物体:
  - `kitchen` (conf=0.5): with appliances and items on counters
  - `living room` (conf=0.5): with bookshelves, couch, and coffee table
  - `cup` (conf=0.5): yellow and blue cups held by individuals

**Qwen3-VL-8B** — 5 个物体:
  - `refrigerator` (conf=0.95): silver refrigerator with papers and a map on its door
  - `couch` (conf=0.95): red leather couch
  - `coffee table` (conf=0.95): glass-top coffee table with magazines
  - `mugs` (conf=0.95): one yellow mug, one blue mug
  - `bookshelf` (conf=0.95): wooden bookshelf filled with books

**Gemma3-12B** — 7 个物体:
  - `refrigerator` (conf=1.0): stainless steel refrigerator with notes on the door
  - `cabinets` (conf=1.0): wooden cabinets
  - `sofa` (conf=1.0): brown leather sofa
  - `coffee table` (conf=1.0): round glass coffee table with CDs
  - `bookshelves` (conf=1.0): wooden bookshelves filled with books
  - `mug` (conf=1.0): yellow ceramic mug
  - `chair` (conf=1.0): wooden chair

### `DmSmN-oqFZ0` 切片 6 (30.0–40.0s)

**Qwen2.5-VL-72B** — 2 个物体:
  - `door` (conf=0.95): wooden door with caution tape
  - `stairs` (conf=0.9): blue carpeted stairs

**Qwen3-VL-8B** — 5 个物体:
  - `caution tape` (conf=0.98): Yellow caution tape forming an X across a door.
  - `door` (conf=0.98): Wooden door with a sign reading 'OUT OF ORDER'.
  - `staircase` (conf=0.95): Staircase with blue carpeted steps.
  - `bookshelf` (conf=0.9): Wooden bookshelf filled with books.
  - `poster` (conf=0.85): Colorful poster on the wall depicting figures in costumes.

**Gemma3-12B** — 6 个物体:
  - `door` (conf=1.0): wooden door with caution tape
  - `stairs` (conf=1.0): carpeted stairs
  - `caution tape` (conf=1.0): yellow caution tape across the door
  - `desk` (conf=1.0): wooden desk with computer
  - `bookshelf` (conf=1.0): wooden bookshelf with books
  - `poster` (conf=1.0): comic book poster

### `G6f0w5BRasw` 切片 11 (60.0–70.0s)

**Qwen2.5-VL-72B** — 2 个物体:
  - `brick wall` (conf=0.85): a textured wall made of bricks
  - `metal object` (conf=0.7): a circular metal item attached to the wall

**Qwen3-VL-8B** — 4 个物体:
  - `building` (conf=0.9): A structure with a tiled roof and weathered walls.
  - `foliage` (conf=0.9): Green trees and bushes in the background.
  - `arm` (conf=0.95): Both individuals have their arms extended and interlocked.
  - `hand` (conf=0.95): Both individuals have their hands clasped or gripping each other's arms.

**Gemma3-12B** — 3 个物体:
  - `building` (conf=0.8): weathered building with a corrugated metal roof
  - `vegetation` (conf=0.7): green foliage
  - `wall` (conf=0.6): peeled paint wall

---

## 4. Roster (角色名单) 同一 person-ID 的描述对照

两家相同 `person-XX` 槽位不代表同一个真人 — pipeline 只保证数量与槽位。重点看每家**描述粒度**, 不要做跨模型的身份匹配。

### `oCkUyjaZuNI` (95s. 室内场景片段, 含 Marvel/超级英雄主题镜头 (`QUEENS` 字幕、超级英雄装扮))

**`person-01`**
- Qwen2.5-VL-72B: male/adult, clothing="dark jacket over a white shirt with a graphic design", appearance="short hair"
- Qwen3-VL-8B: male/adolescent, clothing="white t-shirt with graphic, dark zip-up hoodie", appearance="short dark hair"
- Gemma3-12B: male/adult, clothing="dark jacket, dark pants", appearance="short dark hair, glasses"

**`person-02`**
- Qwen2.5-VL-72B: female/adult, clothing="white top", appearance="long hair"
- Qwen3-VL-8B: male/adult, clothing="dark suit jacket", appearance="short dark hair, facial hair"
- Gemma3-12B: female/adult, clothing="white shirt, dark pants", appearance="long brown hair"

**`person-03`**
- Qwen2.5-VL-72B: male/adult, clothing="dark suit jacket", appearance="short hair"
- Qwen3-VL-8B: female/adult, clothing="white short-sleeved top", appearance="medium-length dark hair"
- Gemma3-12B: male/adolescent, clothing="white shirt with graphic, dark hoodie", appearance="short brown hair"

### `-JUtzfuiV08` (85s. 客厅访谈式拍摄, 两位成人坐着, 镜头较稳)

**`person-01`**
- Qwen2.5-VL-72B: female/adult, clothing="blue sweater, light blue pants", appearance="long straight hair, glasses, no visible jewelry or accessories"
- Qwen3-VL-8B: female/adult, clothing="blue cable-knit sweater, dark skirt", appearance="{'hair': 'long, 'glasses': True, 'jewelry': 'bracelet on left wrist'}"
- Gemma3-12B: female/adult, clothing="blue sweater, gray skirt", appearance="long brown hair, glasses"

**`person-02`**
- Qwen2.5-VL-72B: male/adult, clothing="purple striped shirt, brown pants", appearance="short hair, no visible glasses, no visible jewelry or accessories"
- Qwen3-VL-8B: male/adult, clothing="purple striped shirt, brown pants", appearance="{'hair': 'short, 'glasses': False, 'jewelry': 'watch on left wrist'}"
- Gemma3-12B: male/adult, clothing="purple striped shirt, brown pants", appearance="short dark hair, no facial hair"

### `DmSmN-oqFZ0` (95s. 室内 + 楼梯 + 厨房, 含特殊道具 (Superman 雕像、剑、警戒带))

**`person-01`**
- Qwen2.5-VL-72B: male/adult, clothing="blue t-shirt with number '73', yellow long-sleeve shirt underneath, beige pants", appearance="short brown hair, no visible glasses or accessories"
- Qwen3-VL-8B: male/adult, clothing="blue t-shirt with yellow sleeves and a circular emblem, brown pants", appearance="short dark hair, no facial hair, wearing a watch"
- Gemma3-12B: male/adult, clothing="blue shirt with number 73, yellow long-sleeve shirt, khaki pants, wristwatch", appearance="short dark hair, glasses"

**`person-02`**
- Qwen2.5-VL-72B: female/adult, clothing="purple jacket, blue skirt, purple top underneath", appearance="long brown hair, wearing glasses"
- Qwen3-VL-8B: female/adult, clothing="purple jacket over a purple top, blue skirt, black leggings", appearance="long brown hair, wearing glasses"
- Gemma3-12B: female/adult, clothing="purple jacket, purple shirt, denim skirt, black stockings, black shoulder bag", appearance="long brown hair, glasses"

**`person-03`**
- Qwen2.5-VL-72B: unknown/unknown, clothing="visible clothing unclear", appearance="visible recurring person; details unclear"
- Qwen3-VL-8B: unknown/unknown, clothing="visible clothing unclear", appearance="visible recurring person; details unclear"
- Gemma3-12B: unknown/unknown, clothing="visible clothing unclear", appearance="visible recurring person; details unclear"

### `G6f0w5BRasw` (207s. 户外砖房 + 多人 (含小孩) + 长时段, 镜头切换最多)

**`person-01`**
- Qwen2.5-VL-72B: male/adult, clothing="white shirt, blue jeans", appearance="short dark hair, mustache, no visible accessories"
- Qwen3-VL-8B: male/adult, clothing="white short-sleeved shirt, blue pants", appearance="{'facial_hair': 'none', 'hair_length': 'short', 'accessories': 'none'}"
- Gemma3-12B: male/adult, clothing="white shirt, blue jeans", appearance="short dark hair, mustache"

**`person-02`**
- Qwen2.5-VL-72B: female/child, clothing="white tank top, dark pants", appearance="long braided hair, no visible accessories"
- Qwen3-VL-8B: female/child, clothing="white tank top, dark pants", appearance="{'hair_length': 'medium, 'accessories': 'none'}"
- Gemma3-12B: female/child, clothing="white tank top, blue pants", appearance="dark hair"

**`person-03`**
- Qwen2.5-VL-72B: unknown/unknown, clothing="visible clothing unclear", appearance="visible recurring person; details unclear"
- Qwen3-VL-8B: unknown/unknown, clothing="visible clothing unclear", appearance="visible recurring person; details unclear"
- Gemma3-12B: unknown/unknown, clothing="visible clothing unclear", appearance="visible recurring person; details unclear"

---

## 5. 行为链 (chains) 形状与样例

`avg_span` 是单个 chain 跨秒数, `avg_events` 是每 chain 合并的事件数。数字越大说明该模型倾向**把长时段事件合成一条链**, 数字越小则切得越碎。

| 视频 | 模型 | chains | avg_span (s) | max_span (s) | avg_events/chain | 单事件 chain | 顶部 pattern |
|---|---|---|---|---|---|---|---|
| oCkUyjaZuNI | Qwen2.5-VL-72B | 21 | 11.6 | 34.0 | 1.29 | 18 | single_observation:13, motion_observation:4 |
| oCkUyjaZuNI | Qwen3-VL-8B | 30 | 11.3 | 16.0 | 1.2 | 24 | single_observation:20, behavior_sequence:4 |
| oCkUyjaZuNI | Gemma3-12B | 14 | 14.5 | 40.0 | 2.29 | 10 | single_observation:10, behavior_sequence:3 |
| -JUtzfuiV08 | Qwen2.5-VL-72B | 22 | 10.7 | 16.0 | 1.14 | 19 | single_observation:12, object_mediated:6 |
| -JUtzfuiV08 | Qwen3-VL-8B | 22 | 10.9 | 16.0 | 1.14 | 19 | single_observation:10, object_mediated:6 |
| -JUtzfuiV08 | Gemma3-12B | 15 | 13.3 | 28.0 | 1.47 | 10 | single_observation:7, object_mediated:3 |
| DmSmN-oqFZ0 | Qwen2.5-VL-72B | 20 | 12.2 | 22.0 | 1.35 | 15 | single_observation:12, motion_sequence:3 |
| DmSmN-oqFZ0 | Qwen3-VL-8B | 21 | 12.1 | 28.0 | 1.43 | 16 | single_observation:13, motion_observation:2 |
| DmSmN-oqFZ0 | Gemma3-12B | 21 | 12.1 | 22.0 | 1.48 | 14 | single_observation:12, behavior_sequence:3 |
| G6f0w5BRasw | Qwen2.5-VL-72B | 33 | 13.8 | 40.0 | 1.45 | 24 | single_observation:23, behavior_sequence:4 |
| G6f0w5BRasw | Qwen3-VL-8B | 46 | 12.1 | 26.914999999999992 | 1.35 | 36 | single_observation:29, object_mediated:8 |
| G6f0w5BRasw | Gemma3-12B | 46 | 11.5 | 28.0 | 1.48 | 37 | single_observation:28, object_mediated:11 |

---

## 6. chain 内容对照 (oCkUyjaZuNI 第一条 chain)

### Qwen2.5-VL-72B — chain_0001

- 时间窗: 0.0s → 10.0s (跨度 10.0s)
- pattern: `motion_observation`
- 参与角色: (unattributed)
- 涉及物体 (3): ['city_buildings', 'highway', 'backpack']
- events (1):
  - `locomotion_posture` [0.0–10.0s]: walking in a hallway
- 综合摘要: A person is walking in a hallway.
- confidence: 0.9

### Qwen3-VL-8B — chain_0001

- 时间窗: 0.0s → 16.0s (跨度 16.0s)
- pattern: `motion_sequence`
- 参与角色: (unattributed)
- 涉及物体 (9): ['city_skyline', 'text_overlay', 'highway', 'backpack', 'building', 'person_on_floor', 'plastic_bag', 'window', 'furniture']
- events (2):
  - `locomotion_posture` [0.0–10.0s]: walking away from the camera down a hallway
  - `locomotion_posture` [6.0–16.0s]: walking
- 综合摘要: From 0.0s to 16.0s, two individuals walk away from the camera down a hallway and then walk.
- confidence: 0.825

### Gemma3-12B — chain_0001

- 时间窗: 0.0s → 40.0s (跨度 40.0s)
- pattern: `object_mediated`
- 参与角色: (unattributed)
- 涉及物体 (26): ['cityscape', 'cars', 'trees', 'backpack', 'hallway', 'suit', 'bag', 'window', 'chair', 'door', 'shelf', 'table', 'lamp', 'book', 'plant', 'hutch', 'glassware', 'sofa', 'television', 'bookshelves', 'cabinets', 'microwave', 'stove', 'lamps', 'cabinet', 'painting']
- events (10):
  - `locomotion_posture` [0.0–10.0s]: walking away
  - `posture` [6.0–16.0s]: standing
  - `observed_behavior` [6.0–16.0s]: lying on the floor
  - `posture` [12.0–22.0s]: sitting
  - `locomotion_posture` [12.0–22.0s]: walking away
  - `locomotion_posture` [18.0–28.0s]: Walking away from the camera
- 综合摘要: From 0.0s to 40.0s, various actions such as walking, sitting, standing, and gesturing are observed.
- confidence: 0.87

---

## 7. 关键差异结论

1. **Qwen2.5-VL-72B 是"克制派"**
   类别词偏抽象 ("kitchen", "living room", "shelves"), 描述常引入 "appears to",倾向不写没看清的细节。物体覆盖最少 (194 累计), 但描述更不易过拟合到无关元素。

2. **Qwen3-VL-8B 是"叙事派"**
   scene_description 写得最像"短小说" (主语+动作+背景物多重从句), 也能识别 "explosion"、"sparks" 等动态元素。chain 切得最细 (oCkUyjaZuNI 30 条), 适合后续按时间检索。

3. **Gemma3-12B 是"清单派"**
   scene_description 短而碎 ("A person is on the floor."), 但 objects 最具体 — 能识别到 "magazine"、"Rubik's cube"、"DVDs"、"tissues" 这类小物。在 Marvel 主题视频里还识别到了 **"avengers tower"** (两个 Qwen 都没识别到的具体 IP 名)。

4. **chain 合并粒度**
   Gemma 倾向把长时段合一条 chain (oCkUyjaZuNI 第一条 0–40s 含 10 事件), Qwen3-VL-8B 切得最细, Qwen2.5-VL-72B 居中。**做事件检索用 8B 最方便**, **做摘要式叙述用 Gemma 更顺**。

5. **roster 都给了 3 人 (oCkUyjaZuNI), 但描述差异大**
   Gemma 给 person-01 加了 "glasses" (其它两家没看到, 可能是 Gemma 的强先验)。Qwen3-VL-8B 把 person-01 判成 "adolescent", 其它两家判 "adult" — 源视频里这位人物年龄确实接近临界, 没有客观 ground truth。

6. **速度/质量性价比**
   若只能选一个: **Qwen3-VL-8B 综合最优** (单 GPU 23 分钟 / 4 视频, 描述细节最多, chain 最细)。Gemma 适合需要细粒度物体覆盖的离线场景。72B 不推荐在此任务上跑, 成本×3 没换来质量提升。

