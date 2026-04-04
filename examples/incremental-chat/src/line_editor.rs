//! A minimal cursor-based line editor for raw-mode terminal input.
//!
//! Supports:
//! - **Left/Right arrows** — move cursor within the line
//! - **Home / Ctrl+A** — jump to start of line
//! - **End / Ctrl+E** — jump to end of line
//! - **Backspace** — delete char before cursor
//! - **Delete** — delete char after cursor
//! - **Ctrl+W** — delete word before cursor
//! - **Ctrl+U** — clear entire line
//! - **Ctrl+K** — delete from cursor to end of line
//! - **Alt+Enter / Shift+Enter** — insert newline (multi-line)
//! - **Enter** — submit
//! - **Ctrl+C** — interrupt
//! - **Ctrl+D** — quit (on empty line) or delete char after cursor

use std::io::{self, Write};

use colored::Colorize;

/// A line buffer with cursor position.
pub struct LineEditor {
    /// The full text content (may contain newlines for multi-line).
    buf: String,
    /// Byte offset of the cursor within `buf`.
    cursor: usize,
}

impl LineEditor {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            cursor: 0,
        }
    }

    /// Get the current text.
    pub fn text(&self) -> &str {
        &self.buf
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Clone the buffer text.
    pub fn take_text(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.buf)
    }

    /// Clear the buffer and cursor.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.buf.clear();
        self.cursor = 0;
    }

    /// Insert a character at the cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.buf.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Insert a newline at the cursor.
    pub fn insert_newline(&mut self) {
        self.buf.insert(self.cursor, '\n');
        self.cursor += 1;
    }

    /// Delete the character before the cursor. Returns true if anything changed.
    pub fn backspace(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        // Find the previous char boundary
        let prev = self.buf[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.buf.drain(prev..self.cursor);
        self.cursor = prev;
        true
    }

    /// Delete the character after the cursor. Returns true if anything changed.
    pub fn delete(&mut self) -> bool {
        if self.cursor >= self.buf.len() {
            return false;
        }
        let next = self.cursor + self.buf[self.cursor..].chars().next().map_or(0, |c| c.len_utf8());
        self.buf.drain(self.cursor..next);
        true
    }

    /// Move cursor one character left. Returns true if moved.
    pub fn move_left(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        let prev = self.buf[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.cursor = prev;
        true
    }

    /// Move cursor one character right. Returns true if moved.
    pub fn move_right(&mut self) -> bool {
        if self.cursor >= self.buf.len() {
            return false;
        }
        let next = self.cursor + self.buf[self.cursor..].chars().next().map_or(0, |c| c.len_utf8());
        self.cursor = next;
        true
    }

    /// Move cursor to start of current line.
    pub fn home(&mut self) {
        // Find the start of the current line (after the last \n before cursor)
        self.cursor = self.buf[..self.cursor]
            .rfind('\n')
            .map_or(0, |i| i + 1);
    }

    /// Move cursor to end of current line.
    pub fn end(&mut self) {
        // Find the next \n after cursor, or end of buf
        self.cursor = self.buf[self.cursor..]
            .find('\n')
            .map_or(self.buf.len(), |i| self.cursor + i);
    }

    /// Delete word before cursor (like bash Ctrl+W).
    pub fn delete_word_back(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        let before = &self.buf[..self.cursor];
        // Skip trailing whitespace
        let trimmed_end = before.trim_end().len();
        // Find the last space before that
        let word_start = before[..trimmed_end]
            .rfind(|c: char| c.is_whitespace())
            .map_or(0, |i| i + 1);
        self.buf.drain(word_start..self.cursor);
        self.cursor = word_start;
        true
    }

    /// Delete from cursor to end of line (Ctrl+K).
    pub fn kill_to_end(&mut self) -> bool {
        if self.cursor >= self.buf.len() {
            return false;
        }
        let line_end = self.buf[self.cursor..]
            .find('\n')
            .map_or(self.buf.len(), |i| self.cursor + i);
        if line_end == self.cursor {
            // At a newline — just delete the newline
            self.buf.drain(self.cursor..self.cursor + 1);
        } else {
            self.buf.drain(self.cursor..line_end);
        }
        true
    }

    /// Clear the entire buffer.
    pub fn clear_all(&mut self) -> bool {
        if self.buf.is_empty() {
            return false;
        }
        self.buf.clear();
        self.cursor = 0;
        true
    }

    /// Redraw the entire input on the terminal.
    ///
    /// This erases everything and rewrites the prompt + content with the
    /// cursor at the correct position.
    pub fn redraw(&self) {
        // Count how many display lines the content occupies so we can move up
        let lines: Vec<&str> = self.buf.split('\n').collect();
        let n_lines = lines.len();

        // Move up to the first line (if multi-line)
        if n_lines > 1 {
            // We need to figure out which line the cursor is on
            // and move up from the last displayed line
            print!("\x1b[{}A", n_lines - 1);
        }

        // Rewrite all lines
        for (i, line_text) in lines.iter().enumerate() {
            print!("\r\x1b[2K"); // clear this terminal line
            if i == 0 {
                print!("{} {}", "user>".green(), line_text);
            } else {
                print!("{} {}", "....".bright_black(), line_text);
            }
            if i < n_lines - 1 {
                print!("\r\n");
            }
        }

        // Now position the cursor correctly
        // Find which line and column the cursor is at
        let before_cursor = &self.buf[..self.cursor];
        let cursor_line = before_cursor.matches('\n').count();
        let cursor_col = before_cursor
            .rfind('\n')
            .map_or(before_cursor.len(), |i| before_cursor.len() - i - 1);

        // Move from the last line to the cursor line
        let last_line = n_lines - 1;
        if cursor_line < last_line {
            print!("\x1b[{}A", last_line - cursor_line);
        }

        // Move to the correct column (prompt width + cursor_col)
        let prompt_width = if cursor_line == 0 { 6 } else { 5 }; // "user> " or ".... "
        print!("\r\x1b[{}C", prompt_width + cursor_col);

        let _ = io::stdout().flush();
    }

    /// Check if cursor is at the end (for simple append-only case).
    pub fn cursor_at_end(&self) -> bool {
        self.cursor >= self.buf.len()
    }
}
