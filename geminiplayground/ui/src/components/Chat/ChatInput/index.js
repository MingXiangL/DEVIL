"use client";
import {Textarea} from "@/components/ui/textarea";
import {forwardRef, useEffect, useImperativeHandle, useRef} from "react";
import "@yaireo/tagify/dist/tagify.css";
import "./styles.css";
import Tagify from "@yaireo/tagify";


const isSameDeep = (a, b) => {
    const trans = x => typeof x == 'string' ? x : JSON.stringify(x)
    return trans(a) === trans(b)
}

function tagTemplate(tagData) {
    return (`
        <tag title="${tagData.value}"
                contenteditable='false'
                spellcheck='false'
                tabIndex="-1"
                class='tagify__tag  custom-tag'
                {this.getAttributes(tagData)}>
            <x title='' class='tagify__tag__removeBtn' role='button' aria-label='remove tag'></x>
            <div style="display: flex;  align-items: center"> 
                <div class='tagify__tag__avatar-wrap' >
                    <img  alt="icon" height="32px" width="32px" src="${tagData.icon}" />
                </div>
                <div class='tagify__tag__info' >
                   <span class='tagify__tag-text' style="margin-left: 10px">${tagData.name}</span>
                </div>
            </div>
        </tag>
        `
    )
}

function suggestionItemTemplate(tagData) {
    return `
        <div ${this.getAttributes(tagData)}
            class='tagify__dropdown__item ${tagData.class ? tagData.class : ""}'
            tabindex="0"
            role="option">
            ${tagData.icon ? `
                <div class='tagify__dropdown__item__avatar-wrap'>
                    <img onerror="this.style.visibility='hidden'" height="16" src="${tagData.icon}">
                </div>` : ''
    }
            <strong>${tagData.name}</strong> <br>
            <span>${tagData.description}</span>
        </div>
    `
}


const TagsInput = forwardRef(function TagsInput({settings, whitelist, loading, readOnly, defaultValue, ...rest}, ref) {

    const inputElmRef = useRef()
    const tagifyRef = useRef()

    useEffect(() => {
        if (!inputElmRef.current) return
        if (Tagify === undefined) return
        const tagify = new Tagify(inputElmRef.current, {...settings, ...rest});
        tagifyRef.current = tagify
        return () => {
            tagify.destroy()
            tagifyRef.current = null
        }
    }, [inputElmRef]);

    useImperativeHandle(ref, () => ({
        currentValue: () => {
            //console.log("getMixedTagsAsString", typeof tagifyRef.current.getInputValue())
            return tagifyRef.current.getMixedTagsAsString()
        },
        clear: () => {
            tagifyRef.current.removeAllTags();
        }
    }), [tagifyRef.current, inputElmRef.current])

    useEffect(() => {
        if (tagifyRef.current && loading) {
            tagifyRef.current.loading(true).dropdown.hide()
        }
    }, [loading])

    useEffect(() => {
        const tagify = tagifyRef.current
        if (tagify && defaultValue && !loading) {
            const currentValue = tagify.getInputValue()
            if (!isSameDeep(defaultValue, currentValue)) {
                // console.log("loadOriginalValues", defaultValue);
                // console.log(tagify.parseMixTags(defaultValue));
                // tagify.loadOriginalValues(defaultValue);

                tagify.loadOriginalValues("sssss [[{\"value\":\"roses.jpg\",\"name\":\"roses.jpg\",\"description\":\"\",\"icon\":\"http://localhost:8081/files/thumbnail_roses.jpg\",\"type\":\"image\",\"prefix\":\"@\"}]]")
            }
        }

    }, [defaultValue, loading, whitelist]);


    useEffect(() => {
        const tagify = tagifyRef.current
        if (tagify) {
            tagify.settings.whitelist.length = 0
            // replace whitelist array items
            whitelist && whitelist.length && tagify.settings.whitelist.push(...whitelist)
        }
    }, [whitelist]);

    useEffect(() => {
        const tagify = tagifyRef.current
        if (tagify) {
            tagify.setReadonly(readOnly)
        }
    }, [readOnly])

    return (
        <div className="tags-input">
            <Textarea ref={inputElmRef}
                      className="w-full shadow-none focus-visible:ring-0"/>
        </div>
    )
});


const ChatInputText = forwardRef(function ChatInputText({inputTags, settings, ...rest}, ref) {
    // Tagify settings object
    const baseSettings = {
        maxTags: 20,
        // backspace: "edit",
        mode: "mix",
        keepInvalidTags: true,
        pattern: /@/,
        tagTextProp: 'name',
        pasteAsTags: true,
        // skipInvalid: true,
        enforceWhitelist: true,
        placeholder: "Type your message here...",
        dropdown: {
            enabled: 0, // suggest tags after a single character input
            classname: 'chat-input',
            maxItems: 6,
            mapValueTo: 'name',
            // position: 'text',
            includeSelectedTags: true, // <-- show selected tags in the dropdown
        },
        duplicates: true,
        templates: {
            dropdownItemNoMatch: function (data) {
                return `<div class='${this.settings.classNames.dropdownItem}' value="noMatch" tabindex="0" role="option">No suggestion found for: <strong>${data.value}</strong></div>`
            },
            dropdownItem: suggestionItemTemplate,
            tag: tagTemplate,
        }
    }
    settings = {...baseSettings, ...settings}
    return (<TagsInput ref={ref} settings={settings} whitelist={inputTags} {...rest}/>)
});

export default ChatInputText;