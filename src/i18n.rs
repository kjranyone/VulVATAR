pub fn set_locale(locale: &str) {
    rust_i18n::set_locale(locale);
}

pub fn locale() -> String {
    rust_i18n::locale().to_string()
}

pub fn available_locales() -> Vec<&'static str> {
    vec!["en", "ja", "zh", "ko"]
}

pub fn locale_display_name(code: &str) -> &str {
    match code {
        "en" => "English",
        "ja" => "\u{65e5}\u{672c}\u{8a9e}",
        "zh" => "\u{4e2d}\u{6587}",
        "ko" => "\u{d55c}\u{ad6d}\u{c5b4}",
        _ => code,
    }
}
