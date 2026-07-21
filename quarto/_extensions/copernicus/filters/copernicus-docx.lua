local function trim(s)
  return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function para_text(para)
  return trim(pandoc.utils.stringify(para))
end

local function is_single_image_para(block)
  return block and block.t == "Para" and #block.content == 1 and block.content[1].t == "Image"
end

local function extract_caption(text, prefix)
  local pat = "^" .. prefix .. "%s*:?%s*(.*)$"
  local cap = text:match(pat)
  if cap then
    cap = trim(cap)
    if cap == "" then
      return prefix
    end
    return cap
  end
  return nil
end

function Pandoc(doc)
  if not FORMAT:match("docx") then
    return doc
  end

  local out = {}
  local i = 1
  while i <= #doc.blocks do
    local b = doc.blocks[i]
    local n = doc.blocks[i + 1]

    if is_single_image_para(b) then
      local img = b.content[1]
      local cap = trim(pandoc.utils.stringify(img.caption or {}))

      if n and n.t == "Para" then
        local ntext = para_text(n)
        local from_next = extract_caption(ntext, "Figure")
        if from_next then
          cap = from_next
          i = i + 1
        end
      end

      if cap == "" then
        cap = "Figure"
      end

      img.caption = pandoc.Inlines({ pandoc.Str(cap) })
      table.insert(out, pandoc.Para({ img }))

    elseif b.t == "Table" then
      local cap = ""
      if b.caption and b.caption.long then
        cap = trim(pandoc.utils.stringify(b.caption.long))
      end

      if cap == "" and n and n.t == "Para" then
        local ntext = para_text(n)
        local from_next = extract_caption(ntext, "Table")
        if from_next then
          cap = from_next
          i = i + 1
        end
      end

      if cap ~= "" then
        b.caption = pandoc.Caption(pandoc.Inlines({ pandoc.Str(cap) }))
      end

      table.insert(out, b)

    elseif b.t == "Para" then
      local t = para_text(b)
      if t:match("^Figure%s*:") or t:match("^Table%s*:") then
        -- Skip orphan caption lines that were not attached above.
      else
        table.insert(out, b)
      end

    else
      table.insert(out, b)
    end

    i = i + 1
  end

  doc.blocks = out
  return doc
end
