import {forwardRef, useImperativeHandle, useState} from "react";
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle
} from "@/components/ui/alert-dialog";

export const ConfirmDialog = forwardRef(function ConfirmDialog({...rest}, ref) {
    const [settings, setSettings] = useState({
        title: "",
        message: "",
        isOpen: false,
        onConfirm: null,
        confirmResult: false
    });


    useImperativeHandle(ref, () => ({
        open: (title, message, onConfirm) => {
            setSettings(prev => ({
                ...prev,
                title,
                message,
                onConfirm,
                isOpen: true
            }))
        },
        close: () => {
            setSettings(prev => ({
                ...prev,
                isOpen: false
            }));
        }
    }), [])

    const {title, message, isOpen} = settings

    return (
        <AlertDialog open={isOpen} onOpenChange={(status) => setSettings({
            ...settings,
            isOpen: status
        })} {...rest}>
            <AlertDialogContent>
                <AlertDialogHeader>
                    <AlertDialogTitle>{title}</AlertDialogTitle>
                    <AlertDialogDescription>
                        {message}
                    </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogCancel onClick={() => {
                        setSettings(prev => ({
                            ...prev,
                            isOpen: false
                        }))
                        console.log("cancel")
                        settings.onConfirm(false);
                    }}>
                        Cancel
                    </AlertDialogCancel>
                    <AlertDialogAction onClick={() => {
                        setSettings(prev => ({
                            ...prev,
                            isOpen: false
                        }))
                        console.log("confirm")
                        settings.onConfirm(true);
                    }}>
                        Confirm
                    </AlertDialogAction>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>
    );
});

export default ConfirmDialog;