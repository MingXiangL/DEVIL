"use client";
import React, {useState} from "react";
import './styles.css';
import {icons} from 'lucide-react';

export default function CodeCopyBtn({children}) {
    const [copyOk, setCopyOk] = useState(false);

    const iconColor = copyOk ? '#0af20a' : '#ddd';
    const icon = copyOk ? 'Check' : 'Copy';

    const handleClick = (e) => {
        const code = children.props.children;
        // copy to clipboard
        navigator.clipboard.writeText(code);

        setCopyOk(true);
        setTimeout(() => {
            setCopyOk(false);
        }, 500);
    }

    const LucideIcon = icons[icon];
    return (
        <div className="code-copy-btn">
            <LucideIcon onClick={handleClick} size={18} color={iconColor}/>
        </div>
    )
}